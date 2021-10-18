import geopy.distance as d
import numpy as np
import math

###   Para gerar uma nova matriz basta editar os valores I e J abaixo
###   Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
I,J = 12,15

pTx = [20,25,30,35,40] # Potências de Tx avaliadas

Mij = np.loadtxt('matriz_M{}{}.txt'.format(I,J))
Nij = np.loadtxt('matriz_N{}{}.txt'.format(I,J))

# m e n são tuplas dos índices das quadrículas que precisam ser iteradas para gerar a matriz Smnp
m = []
for i in range(len(Mij)):
    for j in range(len(Mij[0])):
        if Mij[i][j] == 1:
            m.append((i,j))

n = []
for i in range(len(Nij)):
    for j in range(len(Nij[0])):
        if Nij[i][j] != 0:
            n.append((i,j))

#########################################################################
# Coordenadas dos centros das quadrículas (Cij) para o cálculo das distâncias #
#########################################################################

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

###   Geração das matrizes de pesos Smnp e de conectividade Cmnp entre eNoodeBs e clientes
Smnp = np.zeros((len(m), len(n), len(pTx)))
Cmnp = np.zeros((len(m), len(n), len(pTx)))
for i in range(len(m)):
    for j in range(len(n)):
        dist = d.GeodesicDistance(Cij[m[i][0]][m[i][1]],Cij[n[j][0]][n[j][1]]).km
        PL = 0 # PathLoss
        ShF = 0 # Shadow Fading
        dBP = 0.22 # Break Point Distance
        fc = 700 # Frequency Carrier = 700MHz (frequência do 4G/LTE do CCOp Mv)
        hBS = 10 # Altura da eNodeB = 10m
        hUT = 1.5 # Altura dos UEs = 1.5m
        h = 5 # Altura média das construções = 5m (definido para o cenário RMa)
        W = 20 # Largura média das ruas = 20m (definido para o cenário RMa)

        LOS = None
        # Probabilidade de LOS (TR-36.814 tabela B.1.2.1-2)
        prob_LOS = 0
        if (dist*1000 <= 10):
            prob_LOS = 1
        else:
            prob_LOS = math.e**(-(dist*1000-10)/1000)

        if (prob_LOS > np.random.rand()):
            ###############
            ###   LOS   ###
            ############### fc em MHz, dist em km
            LOS = True

            if (dist*1000 > 10 and dist <= dBP):
                PL = 20 * math.log10(40 * math.pi * dist * fc / 3) + min(0.03 * h ** 1.72, 10) * math.log10(dist) - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * dist
                ShF = np.random.normal(0, 4.) # normal(media, desviopradrao)

            elif (dist > dBP):
                PL = 20 * math.log10(40 * math.pi * dBP * fc / 3) + min(0.03 * h ** 1.72, 10) * math.log10(dBP) - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * dBP + 40 * math.log10(dist/dBP)
                ShF = np.random.normal(0, 6.)  # normal(media, desviopradrao)

        else:
            ################
            ###   NLOS   ###
            ################ fc em GHz, dist em m
            LOS = False

            if (dist*1000 > 10):
                PL = 161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h/hBS)**2) * math.log10(hBS) + (43.42 - 3.1 * math.log10(hBS)) * (math.log10(dist*1000) - 3) + 20 * math.log10(fc/1000) - (3.2 * (math.log10(11.75 * hUT))**2 - 4.97)
                ShF = np.random.normal(0, 8.)  # normal(media, desviopradrao)

        for k in range(len(pTx)):
            # pRx = pTx + GeNB + GRx - PL - ShF --> pRx = pTx[k] + 13 - PL - ShF
            pRx = 0
            pRx = pTx[k] + 13 - PL - ShF
            if (pRx > -80):
                Smnp[i][j][k] = 3
                Cmnp[i][j][k] = 1
                print("***   Peso 3   *** pRx: {}".format(pRx))
            elif (pRx > -90):
                Smnp[i][j][k] = 2
                Cmnp[i][j][k] = 1
                print("***   Peso 2   *** pRx: {}".format(pRx))
            elif (pRx > -100):
                Smnp[i][j][k] = 1
                Cmnp[i][j][k] = 1
                print("***   Peso 1   *** pRx: {}".format(pRx))
            else:
                print("***   Peso 0   *** pRx: {}".format(pRx))

        if LOS == True:
            print("LOS: d={}, PL={}, ShF={}".format(dist, PL, ShF))
        else:
            print("NLOS: d={}, PL={}, ShF={}".format(dist, PL, ShF))
        print("Smnp[{}][{}]: {}".format(i,j,Smnp[i][j]))



###   Geração da matriz de conectividade A entre eNoodeBs
A = np.zeros((len(m), len(m), len(pTx), len(pTx))) # Shape (m1,m2,p1,p2)
for i in range(len(m)):
    for j in range(len(m)):
        dist = d.GeodesicDistance(Cij[m[i][0]][m[i][1]],Cij[m[j][0]][m[j][1]]).km
        PL = 0 # PathLoss
        ShF = 0 # Shadow Fading
        dBP = 1.466 # Break Point Distance
        fc = 700 # Frequency Carrier = 700MHz (frequência do 4G/LTE do CCOp Mv)
        hBS = 10 # Altura da eNodeB = 10m
        hUT = 10 # Altura dos UE = neste caso UE = eNodeB = 10m
        h = 5 # Altura média das construções = 5m (definido para o cenário RMa)
        W = 20 # Largura média das ruas = 20m (definido para o cenário RMa)

        LOS = None
        # Probabilidade de LOS (TR-36.814 tabela B.1.2.1-2)
        prob_LOS = 0
        if (dist*1000 <= 10):
            prob_LOS = 1
        else:
            prob_LOS = math.e**(-(dist*1000-10)/1000)

        if (prob_LOS > np.random.rand()):
            ###############
            ###   LOS   ###
            ############### fc em MHz, dist em km
            LOS = True

            if (dist*1000 > 10 and dist <= dBP):
                PL = 20 * math.log10(40 * math.pi * dist * fc / 3) + min(0.03 * h ** 1.72, 10) * math.log10(dist) - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * dist
                ShF = np.random.normal(0, 4.) # normal(media, desviopradrao)

            elif (dist > dBP):
                PL = 20 * math.log10(40 * math.pi * dBP * fc / 3) + min(0.03 * h ** 1.72, 10) * math.log10(dBP) - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * dBP + 40 * math.log10(dist/dBP)
                ShF = np.random.normal(0, 6.)  # normal(media, desviopradrao)

        else:
            ################
            ###   NLOS   ###
            ################ fc em GHz, dist em m
            LOS = False

            if (dist*1000 > 10):
                PL = 161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h/hBS)**2) * math.log10(hBS) + (43.42 - 3.1 * math.log10(hBS)) * (math.log10(dist*1000) - 3) + 20 * math.log10(fc/1000) - (3.2 * (math.log10(11.75 * hUT))**2 - 4.97)
                ShF = np.random.normal(0, 8.)  # normal(media, desviopradrao)

        for k in range(len(pTx)):
            for l in range(len(pTx)):
                # pRx = pTx + GeNB + GRx - PL - ShF --> pRx = pTx[k] + 13 - PL - ShF
                pRx = 0
                pRx = pTx[k] + 13 - PL - ShF
                if (pRx > -80):
                    A[i][j][k][l] = 1
                    print("***   Peso 3   *** pRx: {}".format(pRx))
                elif (pRx > -90):
                    A[i][j][k][l] = 1
                    print("***   Peso 2   *** pRx: {}".format(pRx))
                elif (pRx > -100):
                    A[i][j][k][l] = 1
                    print("***   Peso 1   *** pRx: {}".format(pRx))
                else:
                    print("***   Peso 0   *** pRx: {}".format(pRx))

        if LOS == True:
            print("LOS: d={}, PL={}, ShF={}".format(dist, PL, ShF))
        else:
            print("NLOS: d={}, PL={}, ShF={}".format(dist, PL, ShF))
        print("A[{}][{}]: {}".format(i,j,A[i][j]))



#####   Salva em ARQUIVO para outras análises   #####
Smnp_reshaped = Smnp.reshape(Smnp.shape[0], -1)
np.savetxt("matriz_Smnp{}{}.txt".format(I,J), Smnp_reshaped, fmt="%d")

Cmnp_reshaped = Cmnp.reshape(Cmnp.shape[0], -1)
np.savetxt("matriz_Cmnp{}{}.txt".format(I,J), Cmnp_reshaped, fmt="%d")

A_reshaped = A.reshape(A.shape[0], -1)
np.savetxt("matriz_A{}{}.txt".format(I,J), A_reshaped, fmt="%d")

#####   Carga do arquivo   #####
loaded_Smnp = np.loadtxt('matriz_Smnp{}{}.txt'.format(I,J))
load_original_Smnp = loaded_Smnp.reshape(
    loaded_Smnp.shape[0], loaded_Smnp.shape[1] // 5, 5) # 5 é o shape da terceira dimensão da matriz Smnp

loaded_Cmnp = np.loadtxt('matriz_Cmnp{}{}.txt'.format(I,J))
load_original_Cmnp = loaded_Cmnp.reshape(
    loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)

loaded_A = np.loadtxt('matriz_A{}{}.txt'.format(I,J))
load_original_A = loaded_A.reshape(
    loaded_A.shape[0], loaded_A.shape[0], 5, 5)

# testes
print("Testes da matriz Smnp:")
print("shape do Smnp origem: ", Smnp.shape)
print("shape da matriz carregada do arquivo: ", load_original_Smnp.shape)
if (load_original_Smnp == Smnp).all():
    print("Ok, as duas matrizes são idênticas!")
else:
    print("Erro, as matrizes são diferentes.")

print()
print("Testes da matriz Cmnp:")
print("shape do Smnp origem: ", Cmnp.shape)
print("shape da matriz carregada do arquivo: ", load_original_Cmnp.shape)
if (load_original_Cmnp == Cmnp).all():
    print("Ok, as duas matrizes são idênticas!")
else:
    print("Erro, as matrizes são diferentes.")

print()
print("Testes da matriz A:")
print("shape do A origem: ", A.shape)
print("shape da matriz carregada do arquivo: ", load_original_A.shape)
if (load_original_A == A).all():
    print("Ok, as duas matrizes são idênticas!")
else:
    print("Erro, as matrizes são diferentes.")