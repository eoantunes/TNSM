from sklearn.cluster import KMeans
import numpy as np
import geopy.distance as d
import math
import time

I,J = 12,15     # Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
k = 1           # Nr de clusters e consequentemente eNodeBs

o = np.loadtxt('result/Exato{}{}_i{}c{}.txt'.format(I,J,Interc,consensoCobertura))
otimo = o[1]
Usu = 100  # Nr máximo de usuários associados a uma eNodeB

def ordem(n):
    return n[2]

Mij = np.loadtxt('matriz/M{}{}.txt'.format(I,J), dtype=int)
Nij = np.loadtxt('matriz/N{}{}.txt'.format(I,J), dtype=int)
loaded_Cmnp = np.loadtxt('matriz/Cmnp{}{}.txt'.format(I,J), dtype=int)
Cmnp = loaded_Cmnp.reshape(loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)

# m e n são tuplas dos índices das quadrículas que precisam ser iteradas
mm = []
for i in range(len(Mij)):
    for j in range(len(Mij[0])):
        if Mij[i][j] == 1:
            mm.append((i,j))

nn = []
for i in range(len(Nij)):
    for j in range(len(Nij[0])):
        if Nij[i][j] != 0:
            nn.append((i,j))
N = len(nn)
M = len(mm)
P = 5

iteracoes = 30
# time, bFit, nr_eNodeBs, nr_cliAtendidos, areaCoberta, sobrePos2, sobrePos3, sobrePos4, sobrePos5, sobrePos6, sobrePos7, sobrePos8, sobrePos9, PRD
result = np.zeros((iteracoes, 14))
for it in range(iteracoes):
    antes = time.time()
    kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300)
    kmeans.fit(nn)
    centroides = kmeans.cluster_centers_ # Centróides retornados pelo algoritmo Kmeans (valores em float)
    centr = np.zeros_like(centroides)
    for i in range(len(centroides)):
        for j in range(len(centroides[0])):
            centr[i][j] = round(centroides[i][j]) # Arredonda os índices (i,j) obtidos pelo Kmeas para o inteiro mais próximo

    eNodeBs = []
    for c in centr:
        M_IJdist = [] # (i,j,dist)
        for m in range(len(mm)):
            M_IJdist.append( (mm[m][0], mm[m][1], math.sqrt((mm[m][0] - c[0])*(mm[m][0] - c[0]) + (mm[m][1] - c[1])*(mm[m][1] - c[1])) ) )
        M_IJdist.sort(key=ordem)
        eNodeBs.append((M_IJdist[0][0], M_IJdist[0][1])) # Faz a associação à posição de eNodeB (Mij) mais próxima aos clusters obtidos em  Nij
    print(eNodeBs)

    ###################################################
    # Avaliando a solução e armazenando suas métricas #
    ###################################################
    idx_eNodeBs = []
    for m in range(M):
        if mm[m] in eNodeBs:
            idx_eNodeBs.append(m)

    result[it][0] = time.time() - antes                                             # time

    grauInterf = np.zeros((M, P))  # Medida aproximada do impacto de cada eNodeB sobre a interferência total (aproximada, pois a medida exata é dependente do conjunto de eNodeBs ativadas em cada solução)
    for m in range(0, M):
        for p in range(0, P):
            aux = 0
            for n in range(0, N):
                aux += Cmnp[m][n][p]
            grauInterf[m][p] = aux / N

    interf = 0
    for m in idx_eNodeBs:
        interf += grauInterf[m][2]

    result[it][1] = -1 / 8 * interf                                                 # bFit

    result[it][2] = k                                                               # nr_eNodeBs

    # nr_CliAtendidos
    cliNaoAtendidos = np.zeros(N)
    for n in range(N):
        cliNaoAtendidos[n] = Nij[nn[n][0]][nn[n][1]]

    for m in range(M):
        aux = 0  # Nr de clientes atendidos por eNodeB
        quadriculasAtendidas = []
        for n in range(N):
            if aux < Usu and m == idx_eNodeBs and Cmnp[m][n][2] == 1 and cliNaoAtendidos[n] > 0:
                cliAux = cliNaoAtendidos[n]
                aux1 = aux
                aux += cliNaoAtendidos[n]
                cliNaoAtendidos[n] = 0
                quadriculasAtendidas.append(n)
                if aux > Usu:
                    cliNaoAtendidos[n] = cliAux
                    aux = aux1
        if aux != 0:
            print("Clientes atendidos pela eNodeB [{},2] = {}".format(m, aux))

    nrTotalClientes = 0
    nrClientesNaoAtendidos = 0
    for n in range(N):
        nrTotalClientes += Nij[nn[n][0]][nn[n][1]]
        nrClientesNaoAtendidos += cliNaoAtendidos[n]

    result[it][3] = nrTotalClientes - nrClientesNaoAtendidos                        # nr_cliAtendidos

    ###############################################################################
    # Coordenadas dos centros das quadrículas (Cij) para o cálculo das distâncias #
    ###############################################################################
    ###   Delimitadores da área de atuação   ###
    ###   AD                                 ###
    ###   BC                                 ###
    ###   (lat, long)
    AA = (-20.115437, -44.16269)
    BB = (-20.159446, -44.16269)
    DD = (-20.115437, -44.10444)

    dl = (AA[0] - BB[0]) / I
    dc = (DD[1] - AA[1]) / J

    c00 = (AA[0] - dl / 2, AA[1] + dc / 2)

    Cij = np.zeros((I, J)).tolist()
    for i in range(I):
        for j in range(J):
            Cij[i][j] = (c00[0] - i * dl, c00[1] + j * dc)
            # distância em km entre os centros das quadrículas 00 e 01 = d.GeodesicDistance(Cij[0][0],Cij[0][1]).km

    Cobertura = np.zeros((I, J), dtype=int)
    ##########      Parâmetros para o cálculo do Path Loss      ##########
    dBP = 0.22  # Break Point Distance
    fc = 700  # Frequency Carrier = 700MHz (frequência do 4G/LTE do CCOp Mv)
    hBS = 10  # Altura da eNodeB = 10m
    hUT = 1.5  # Altura dos UEs = 1.5m
    h = 5  # Altura média das construções = 5m (definido para o cenário RMa)
    W = 20  # Largura média das ruas = 20m (definido para o cenário RMa)
    #######################################################################
    for eNB in idx_eNodeBs:
        mi = mm[eNB][0]
        mj = mm[eNB][1]
        for i in range(I):
            for j in range(J):
                dist = d.GeodesicDistance(Cij[mi][mj], Cij[i][j]).km
                # Path Loss (PL) para RMa NLOS
                if dist > 0.01:
                    PL = 161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h / hBS) ** 2) * math.log10(hBS) + (43.42 - 3.1 * math.log10(hBS)) * (math.log10(dist * 1000) - 3) + 20 * math.log10(fc / 1000) - (3.2 * (math.log10(11.75 * hUT)) ** 2 - 4.97)
                else:
                    PL = 0
                pRx = 30 + 13 - PL
                if (pRx > -100):
                    Cobertura[i][j] += 1
    #print(Cobertura)

    areaDaQuadricula = d.GeodesicDistance(Cij[0][0], Cij[0][1]).km
    areaDaQuadricula *= areaDaQuadricula

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

    result[it][4] = quadCobertas * areaDaQuadricula                                 # areaCoberta

    result[it][5] = quadSobrepostas2 * areaDaQuadricula                             # sobrePos2
    result[it][6] = quadSobrepostas3 * areaDaQuadricula                             # sobrePos3
    result[it][7] = quadSobrepostas4 * areaDaQuadricula                             # sobrePos4
    result[it][8] = quadSobrepostas5 * areaDaQuadricula                             # sobrePos5
    result[it][9] = quadSobrepostas6 * areaDaQuadricula                             # sobrePos6
    result[it][10] = quadSobrepostas7 * areaDaQuadricula                            # sobrePos7
    result[it][11] = quadSobrepostas8 * areaDaQuadricula                            # sobrePos8
    result[it][12] = quadSobrepostas9 * areaDaQuadricula                            # sobrePos9

    result[it][13] = 100 * (otimo - result[it][1]) / abs(otimo)                     # PRD

result = np.asarray(result)
np.savetxt('result/Heur{}{}_k{}.txt'.format(I,J,k), result)