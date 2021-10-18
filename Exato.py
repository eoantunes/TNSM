from ortools.linear_solver import pywraplp
import numpy as np
import time

###   Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
I,J = 12,15

def m2a(i,j,line_length):
    return i*line_length + j

# Declaração do solver ORTools considerando a biblioteca CBC do Google
solver = pywraplp.Solver("Exato", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# registro do momento do início do processamento
antes = time.time()



##################
# Parametrização #
##################

#####################
###   Matrizes    ###
#####################
Mij = np.loadtxt('matriz_M{}{}.txt'.format(I,J))
Nij = np.loadtxt('matriz_N{}{}.txt'.format(I,J))

loaded_Smnp = np.loadtxt('matriz_Smnp{}{}.txt'.format(I,J))
Smnp = loaded_Smnp.reshape(loaded_Smnp.shape[0], loaded_Smnp.shape[1] // 5, 5)

loaded_Cmnp = np.loadtxt('matriz_Cmnp{}{}.txt'.format(I,J))
Cmnp = loaded_Cmnp.reshape(loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)

loaded_A = np.loadtxt('matriz_A{}{}.txt'.format(I,J))
A = loaded_A.reshape(loaded_A.shape[0], loaded_A.shape[0], 5, 5)

# Tuplas das quadrículas aptas a receber um eNodeB
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

##############################
M = Smnp.shape[0]  # Nr de pontos de demanda
N = Smnp.shape[1]  # Nr de possíveis locais para instalação de eNodeBs
P = Smnp.shape[2]  # Nr de potências sendo avaliadas
Ant = 9  # Nr máximo de antenas (No CCOp Mv o número máximo é 9 = 8 Vtr Nó de acesso + 1 Centro de Coordenação)
Usu = 100  # Nr máximo de usuários associados a uma eNodeB
Interc = 1 # Nr mínimo de nós interconectados



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
for n in range(0, N):
    ct = solver.Constraint(0, Ant, str(head))
    ct.SetCoefficient(X[n], -1)
    head += 1
    for m in range(0, M):
        for p in range(0, P):
            ct.SetCoefficient(Y[m2a(m, p, P)], int(Cmnp[m][n][p]))

# Restrição (2): Número de usuários por eNodeB deve ser menor ou igual a Usu
#for m in range(0, M):
#    ct = solver.Constraint(0, Usu, str(head))
#    head += 1
#    for n in range(0, N):
#        for p in range(0, P):
            #ct.SetCoefficient(X[n]*Y[m2a(m, p, P)], int(Cmnp[m][n][p]))
#            ct.SetCoefficient(X[n], int(Cmnp[m][n][p] * Nij[nn[n][0]][nn[n][1]]))

# Cada quadrícula de cliente Nij só pode ser atendida ou associada a uma eNodeB
for n in range(0, N):
    ct = solver.Constraint(0, 0, str(head))
    ct.SetCoefficient(X[n], -1)
    head += 1
    for m in range(0, M):
        ct.SetCoefficient(Z[m2a(m, n, N)], 1)

# Cada eNodeB não pode atender ou ter mais clientes associados que sua capacidade máxima = Usu
for m in range(0, M):
    ct = solver.Constraint(0, Usu, str(head))
    head += 1
    for n in range(0, N):
        ct.SetCoefficient(Z[m2a(m, n, N)], Nij[nn[n][0]][nn[n][1]])


# Restrição (3): Número máximo de antenas instaladas ==> Somatório (mEM) Somatório (pEP) ymp <= Ant
ct = solver.Constraint(Interc, Ant, str(head))
for m in range(0, M):
    for p in range(0, P):
        ct.SetCoefficient(Y[m2a(m, p, P)], 1)
head += 1

# Restrição (4): Número mínimo de antenas interconectadas - ROA 65
for m in range(0, M):
    for p in range(0, P):
        ct = solver.Constraint(Interc, Ant, str(head))
        ct.SetCoefficient(Y[m2a(m, p, P)], -1)
        head += 1
        for k in range(0, M):
            for z in range(0, P):
                ct.SetCoefficient(Y[m2a(k, z, P)], int(A[m][k][p][z]))

# Restrição (5): Uma eNodeB só pode ter uma potência de transmissão (pTx) ativada
for m in range(0, M):
    ct = solver.Constraint(0, 1, str(head))
    head += 1
    for p in range(0, P):
        ct.SetCoefficient(Y[m2a(m, p, P)], 1)

###################
# Função objetivo #
###################

objetivo = solver.Objective()

# Maximização do número de clientes atendidos ou cobertura
for n in range(0, N):
    #objetivo.SetCoefficient(X[n], maior(multVector(Y, C[i])) * Nij[nn[n][0]][nn[n][1]]) # Voltar para analisar a função top
    objetivo.SetCoefficient(X[n], int(Nij[nn[n][0]][nn[n][1]]))

# Minimização do número de eNodeBs empregadas
for m in range (0, M):
    for p in range(0, P):
        objetivo.SetCoefficient(Y[m2a(m, p, P)], -1)

# Minimização do grau de interferência
for m in range(0, M):
    for n in range(0, N):
        for p in range(0, P):
            objetivo.SetCoefficient(Y[m2a(m, p, P)], -1 * Cmnp[m][n][p])
for n in range(0, N):
    objetivo.SetCoefficient(X[n], 1)

# Minimização do custo de associação
#for m in range(0, M):
#    for n in range(0, N):
#        objetivo.SetCoefficient(Z[m2a(m, n, N)], -1)

objetivo.SetMaximization()
solver.Solve()
depois = time.time()

#################################
# Imprimindo a solução no shell #
#################################
print()
print('Valor objetivo =', objetivo.Value())
print()

NrClientesAtendidos = 0
NrQuadriculasCobertas = 0
Xstrmatrix = '[ '
for n in range(0, N):
    NrQuadriculasCobertas += int(X[n].solution_value())
    NrClientesAtendidos += int(X[n].solution_value()) * Nij[nn[n][0]][nn[n][1]]
    Xstrmatrix += str(int(X[n].solution_value())) + ' '
Xstrmatrix += ']\n'
print("Cobertura: {}%".format(100*NrQuadriculasCobertas/Smnp.shape[1]))
print('{} quadrículas cobertas de um total de {} quadrículas com clientes.'.format(NrQuadriculasCobertas, Smnp.shape[1]))
print(str(int(NrClientesAtendidos)) + ' clientes atendidos.')
print("X " + Xstrmatrix)

NrAntenasInstaladas = 0
for m in range(0, M):
    for p in range(0, P):
        NrAntenasInstaladas += Y[m2a(m,p,P)].solution_value()
        if int(Y[m2a(m,p,P)].solution_value()) == 1:
            print(Y[m2a(m,p,P)])

print(str(int(NrAntenasInstaladas)) + ' eNodeBs instalados.')

print()
print("Associações de quadrículas de eNodeBs Mij e quadrículas de clientes Nij:")
for m in range(0, M):
    for n in range(0, N):
        if Z[m2a(m, n, N)] == 1:
            print(Z[m2a(m, n, N)])
        else:
            print("Opa, apareceu Zmn = 0!")


print()
print('Tempo gasto no processamento (time) = ' + str(depois - antes))
print("Tempo de processamento = %f" % solver.wall_time())
print("Iterações: %d" % solver.iterations())
print("Problema resolvido usando %d branch-and-bound nós" % solver.nodes())

