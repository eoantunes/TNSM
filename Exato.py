from ortools.linear_solver import pywraplp
import numpy as np
import geopy.distance as d
import math

###   Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
I,J = 12,15

def m2a(i,j,line_length):
    return i*line_length + j

# Declaração do solver ORTools considerando a biblioteca CBC do Google
solver = pywraplp.Solver("Exato", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

##################
# Parametrização #
##################

# Matrizes
Mij = np.loadtxt('matriz/M{}{}.txt'.format(I,J))
Nij = np.loadtxt('matriz/N{}{}.txt'.format(I,J))
#loaded_Smnp = np.loadtxt('matriz/Smnp{}{}.txt'.format(I,J))
#Smnp = loaded_Smnp.reshape(loaded_Smnp.shape[0], loaded_Smnp.shape[1] // 5, 5)
loaded_Cmnp = np.loadtxt('matriz/Cmnp{}{}.txt'.format(I,J))
Cmnp = loaded_Cmnp.reshape(loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)
A = np.loadtxt('matriz/A{}{}.txt'.format(I,J))

# Tuplas dos índices das quadrículas aptas a receber um eNodeB
mm = []
for i in range(len(Mij)):
    for j in range(len(Mij[0])):
        if Mij[i][j] == 1:
            mm.append((i,j))
# Tuplas dos índices das quadrículas que possuem clientes
nn = []
for i in range(len(Nij)):
    for j in range(len(Nij[0])):
        if Nij[i][j] != 0:
            nn.append((i,j))

M = Cmnp.shape[0]  # Nr de pontos de demanda
N = Cmnp.shape[1]  # Nr de possíveis locais para instalação de eNodeBs
P = Cmnp.shape[2]  # Nr de potências sendo avaliadas
Ant = 9  # Nr máximo de antenas (No CCOp Mv o número máximo é 9 = 8 Vtr Nó de acesso + 1 Centro de Coordenação)
Usu = 100  # Nr máximo de usuários associados a uma eNodeB
Interc = 1 # Nr mínimo de nós interconectados (uma eNodeB precisa estar conectada a mais Interc eNodeBs)
consensoCobertura = 0.95 # Porcentagem do número de clientes que devem ser atendidos

grauInterf = np.zeros((M,P)) # Medida aproximada do impacto de cada eNodeB sobre a interferência total (aproximada, pois a medida exata é dependente do conjunto de eNodeBs ativadas em cada solução)
for m in range(0, M):
    for p in range(0, P):
        aux = 0
        for n in range(0, N):
            aux += Cmnp[m][n][p]
        grauInterf[m][p] = aux/N


########################
# Variáveis de decisão #
########################

X = []  # clientes
for n in range(0, N):
    X.append(solver.BoolVar('X%d' % (n)))

Y = []  # eNodeBs
for m in range(0, M):
    for p in range(0, P):
        Y.append(solver.BoolVar('Y[%d,%d]'%(m,p)))

Z = []  # Associação de quadrículas de clientes à eNodeBs --> Zmn = 1, se a quadrícula n for designada para ser atendida pela eNodeB m
for m in range(0, M):
    for n in range(0, N):
        Z.append(solver.BoolVar('Z[%d,%d]'%(m,n)))

##############
# Restrições #
##############
head = 0

# Restrição (1):  Somatório (mEM) Somatório (pEP) ymp.cmnp >= xn para todo nEN
#                 Somatório (mEM) Somatório (pEP) ymp.cmnp - xn >= 0 para todo nEN
# Para que um cliente seja atendido (Xn = 1), ao menos um eNodeB que lhe dê cobertura deve ser instalado
for n in range(0, N):
    ct = solver.Constraint(0, Ant, str(head))
    ct.SetCoefficient(X[n], -1)
    head += 1
    for m in range(0, M):
        for p in range(0, P):
            ct.SetCoefficient(Y[m2a(m, p, P)], int(Cmnp[m][n][p]))

# Restrição (2): Cada quadrícula de cliente Nij só pode ser atendida ou associada a um eNodeB
for n in range(0, N):
    ct = solver.Constraint(0, 0, str(head))
    ct.SetCoefficient(X[n], -1)
    head += 1
    for m in range(0, M):
        ct.SetCoefficient(Z[m2a(m, n, N)], 1)

# Restrição (3): Cada eNodeB não pode atender ou ter mais clientes associados que sua capacidade máxima = Usu
for m in range(0, M):
    ct = solver.Constraint(0, Usu, str(head))
    head += 1
    for n in range(0, N):
        ct.SetCoefficient(Z[m2a(m, n, N)], Nij[nn[n][0]][nn[n][1]])

# Restrição (4): As quadrículas de clientes Nij só podem ser associadas a eNodeBs instaladas ou ativadas --> Somatório (pEP) Ymp = 1
# Se uma antena Mij está desativada então não devem haver clientes associados a ela:
for m in range(0, M):
    ct = solver.Constraint(-1000, 0, str(head))
    head += 1
    for n in range(0, N):
        ct.SetCoefficient(Z[m2a(m, n, N)], Nij[nn[n][0]][nn[n][1]])
    for p in range(0, P):
        ct.SetCoefficient(Y[m2a(m, p, P)], -250)

# Restrição (5): Número máximo de eNodeBs instalados ==> Somatório (mEM) Somatório (pEP) ymp <= Ant
ct = solver.Constraint(Interc, Ant, str(head))
for m in range(0, M):
    for p in range(0, P):
        ct.SetCoefficient(Y[m2a(m, p, P)], 1)
head += 1

# Restrição (6): Número mínimo de antenas interconectadas - ROA 65 (Não implementado no GA)
for m in range(0, M):
    for p in range(0, P):
        ct = solver.Constraint(Interc, Ant, str(head))
        ct.SetCoefficient(Y[m2a(m, p, P)], -1)
        head += 1
        for k in range(0, M):
            for l in range(0, P):
                ct.SetCoefficient(Y[m2a(k, l, P)], int(A[m][k]))

# Restrição (7): Uma eNodeB só pode ter uma potência de transmissão (pTx) ativada
for m in range(0, M):
    ct = solver.Constraint(0, 1, str(head))
    head += 1
    for p in range(0, P):
        ct.SetCoefficient(Y[m2a(m, p, P)], 1)

# Restrição da Maximização da Área de Cobertura
ct = solver.Constraint(consensoCobertura, 1, str(head))
head += 1
for n in range(0,N):
    ct.SetCoefficient(X[n], Nij[nn[n][0]][nn[n][1]]/200)

#ct = solver.Constraint(-0.99 ,-0.1 ,str(head))
#head += 1
#for n in range(0,N):
#    ct.SetCoefficient(Y[m2a(m, p, P)], - 1/9)

###################
# Função objetivo #
###################
objetivo = solver.Objective()

# Maximização do número de clientes atendidos ou cobertura
#for n in range(0, N):
#   objetivo.SetCoefficient(X[n], Nij[nn[n][0]][nn[n][1]]/200)  # peso definido pela análise da curva de Paretto

# Minimização do número de eNodeBs empregadas
#                   e
# Minimização do grau de interferência
for m in range(0, M):
    for p in range(0, P): #                      interf                 nr_eNodeBs
        #objetivo.SetCoefficient(Y[m2a(m, p, P)], (-1/9 * grauInterf[m][p]) -1/9)
        objetivo.SetCoefficient(Y[m2a(m, p, P)], -1/8 * grauInterf[m][p] )
        #objetivo.SetCoefficient(Y[m2a(m, p, P)], - 1/9)


objetivo.SetMaximization()
solver.Solve()
#######################################################################################################################
#################################
# Imprimindo a solução no shell #
#################################
print("Número de quadrículas aptas a instalação de um eNodeB = %d" % Cmnp.shape[0])
print("Número de quadrículas com clientes = %d" % Cmnp.shape[1])
NrClientesAtendidos = 0
NrQuadriculasCobertas = 0
NrAssociacoes = 0
Xstrmatrix = '[ '
for n in range(0, N):
    NrQuadriculasCobertas += int(X[n].solution_value())
    NrClientesAtendidos += int(X[n].solution_value()) * Nij[nn[n][0]][nn[n][1]]
    Xstrmatrix += str(int(X[n].solution_value())) + ' '
Xstrmatrix += ']\n'
print("Cobertura: {}% --> {} quadrículas cobertas de um total de {} com clientes.".format(100*NrQuadriculasCobertas/Cmnp.shape[1], NrQuadriculasCobertas, Cmnp.shape[1]))
print(str(int(NrClientesAtendidos)) + ' clientes atendidos.')
print("X " + Xstrmatrix)

sol = np.zeros((M, P))
NrAntenasInstaladas = 0
eNB_pot =[]
for m in range(0, M):
    for p in range(0, P):
        NrAntenasInstaladas += Y[m2a(m,p,P)].solution_value()
        if int(Y[m2a(m,p,P)].solution_value()) == 1:
            print(Y[m2a(m,p,P)])
            aux = str(Y[m2a(m,p,P)])
            aux = aux[2:-1]
            aux = aux.split(",")
            m = int(aux[0])
            p = int(aux[1])
            sol[m][p] = 1
            eNB_pot.append((m,p))

print(str(int(NrAntenasInstaladas)) + ' eNodeBs instalados.')

print()
print("Associações de quadrículas de eNodeBs Mij e quadrículas de clientes Nij:")
for m in range(0, M):
    for n in range(0, N):
        if int(Z[m2a(m, n, N)].solution_value()) == 1:
            NrAssociacoes += 1
            print(Z[m2a(m, n, N)])
print("Número de associações = %d" % NrAssociacoes)

print()
print("Tempo de processamento = %f" % (solver.wall_time()/1000))

###################################################
# Avaliando a solução e armazenando suas métricas #
###################################################
interf = 0
for m in range(M):
    for p in range(P):
        interf += grauInterf[m][p] * sol[m][p]

#valorObjetivo = NrClientesAtendidos/200 - 1/9 * NrAntenasInstaladas - 1/8 * interf
valorObjetivo = - 1/8 * interf

if (objetivo.Value() == valorObjetivo):
    print("O Valor Objetivo avaliado é idêntico ao retornado pelo solver = {}".format(valorObjetivo))
else:
    print("Valor objetivo (solver) = {}".format(objetivo.Value()))
    print("Valor objetivo (cálculo) = {}".format(valorObjetivo))
print("Grau de interferência = %f" % interf)

###############################################################################
# Coordenadas dos centros das quadrículas (Cij) para o cálculo das distâncias #
###############################################################################

###   Delimitadores da área de atuação   ###
###   AD                                 ###
###   BC                                 ###
###   (lat, long)
A = (-20.115437,-44.16269)
B = (-20.159446,-44.16269)
D = (-20.115437,-44.10444)

dl = (A[0]-B[0])/I
dc = (D[1]-A[1])/J

c00 = (A[0]-dl/2, A[1] + dc/2)

Cij = np.zeros((I, J)).tolist()
for i in range(I):
  for j in range(J):
    Cij[i][j] = (c00[0] - i*dl, c00[1] + j*dc)
# distância em km entre os centros das quadrículas 00 e 01 = d.GeodesicDistance(Cij[0][0],Cij[0][1]).km

Cobertura = np.zeros((I,J))
pTx = [20,25,30,35,40] # Potências de Tx avaliadas

##########      Parâmetros para o cálculo do Path Loss      ##########
dBP = 0.22 # Break Point Distance
fc = 700 # Frequency Carrier = 700MHz (frequência do 4G/LTE do CCOp Mv)
hBS = 10 # Altura da eNodeB = 10m
hUT = 1.5 # Altura dos UEs = 1.5m
h = 5 # Altura média das construções = 5m (definido para o cenário RMa)
W = 20 # Largura média das ruas = 20m (definido para o cenário RMa)
#######################################################################
for eNB in eNB_pot:
    m = eNB[0]
    mi = mm[m][0]
    mj = mm[m][1]
    p = pTx[eNB[1]]
    for i in range(I):
        for j in range(J):
            dist = d.GeodesicDistance(Cij[mi][mj],Cij[i][j]).km
            # Path Loss (PL) para RMa NLOS
            if dist > 0.01:
                PL = 161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h / hBS) ** 2) * math.log10(hBS) + (43.42 - 3.1 * math.log10(hBS)) * (math.log10(dist * 1000) - 3) + 20 * math.log10(fc / 1000) - (3.2 * (math.log10(11.75 * hUT)) ** 2 - 4.97)
            else:
                PL = 0
            pRx = p + 13 - PL
            if (pRx > -100):
                Cobertura[i][j] += 1
print(Cobertura)

areaDaQuadricula = d.GeodesicDistance(Cij[0][0],Cij[0][1]).km
areaDaQuadricula *= areaDaQuadricula
print("Área de 1 quadrícula = {} km2".format(areaDaQuadricula))
quadCobertas = 0
quadSobrepostas2 = 0
quadSobrepostas3 = 0
quadSobrepostas4 = 0
quadSobrepostas5 = 0
quadSobrepostas6 = 0
quadSobrepostas7 = 0
quadSobrepostas8 = 0
quadSobrepostas9 = 0
for i in range(I):
    for j in range(J):
        if Cobertura[i][j] > 0:
            quadCobertas += 1
            if Cobertura[i][j] == 2: quadSobrepostas2 += 1
            if Cobertura[i][j] == 3: quadSobrepostas3 += 1
            if Cobertura[i][j] == 4: quadSobrepostas4 += 1
            if Cobertura[i][j] == 5: quadSobrepostas5 += 1
            if Cobertura[i][j] == 6: quadSobrepostas6 += 1
            if Cobertura[i][j] == 7: quadSobrepostas7 += 1
            if Cobertura[i][j] == 8: quadSobrepostas8 += 1
            if Cobertura[i][j] == 9: quadSobrepostas9 += 1
print("Número de quadrículas cobertas = {}".format(quadCobertas))
print("Área coberta = {}".format(quadCobertas*areaDaQuadricula))
print("Sobreposições")
print("Quadrículas com sobreposição de 2 eNB = {} --> área {}".format(quadSobrepostas2, quadSobrepostas2*areaDaQuadricula))
print("Quadrículas com sobreposição de 3 eNB = {} --> área {}".format(quadSobrepostas3, quadSobrepostas3*areaDaQuadricula))
print("Quadrículas com sobreposição de 4 eNB = {} --> área {}".format(quadSobrepostas4, quadSobrepostas4*areaDaQuadricula))
print("Quadrículas com sobreposição de 5 eNB = {} --> área {}".format(quadSobrepostas5, quadSobrepostas5*areaDaQuadricula))
print("Quadrículas com sobreposição de 6 eNB = {} --> área {}".format(quadSobrepostas6, quadSobrepostas6*areaDaQuadricula))
print("Quadrículas com sobreposição de 7 eNB = {} --> área {}".format(quadSobrepostas7, quadSobrepostas7*areaDaQuadricula))
print("Quadrículas com sobreposição de 8 eNB = {} --> área {}".format(quadSobrepostas8, quadSobrepostas8*areaDaQuadricula))
print("Quadrículas com sobreposição de 9 eNB = {} --> área {}".format(quadSobrepostas9, quadSobrepostas9*areaDaQuadricula))