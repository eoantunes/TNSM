import pygad
import numpy as np
import random
import time as t
import geopy.distance as d
import math
import matplotlib.pyplot as plt

I,J = 12,15                 # Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
grauInterc = [0, 1, 2, 3]                  # Nr mínimo de nós interconectados (uma eNodeB precisa estar conectada a mais Interc eNodeBs)(0, 1, 2, 3)
percentualCobertura = [0.8, 0.9, 0.95, 0.98, 1]    # Porcentagem do número de clientes que devem ser atendidos (0.8, 0.9, 0.95, 0.98, 1)

for Interc in grauInterc:
    for consensoCobertura in percentualCobertura:

        o = np.loadtxt('result/Exato{}{}_i{}c{}.txt'.format(I,J,Interc,consensoCobertura))
        otimo = o[1]

        Mij = np.loadtxt('matriz/M{}{}.txt'.format(I,J), dtype=int)
        Nij = np.loadtxt('matriz/N{}{}.txt'.format(I,J), dtype=int)
        loaded_Cmnp = np.loadtxt('matriz/Cmnp{}{}.txt'.format(I,J), dtype=int)
        Cmnp = loaded_Cmnp.reshape(loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)
        A = np.loadtxt('matriz/A{}{}.txt'.format(I,J), dtype=int)

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
        M = Cmnp.shape[0]  # Nr de pontos de demanda
        N = Cmnp.shape[1]  # Nr de possíveis locais para instalação de eNodeBs
        P = Cmnp.shape[2]  # Nr de potências sendo avaliadas
        Ant = 9  # Nr máximo de antenas (No CCOp Mv o número máximo é 9 = 8 Vtr Nó de acesso + 1 Centro de Coordenação)
        Usu = 100  # Nr máximo de usuários associados a uma eNodeB

        grauInterf = np.zeros((M, P)) # Medida aproximada do impacto de cada eNodeB sobre a interferência total (aproximada, pois a medida exata é dependente do conjunto de eNodeBs ativadas em cada solução)
        for m in range(0, M):
            for p in range(0, P):
                aux = 0
                for n in range(0, N):
                    aux += Cmnp[m][n][p]
                grauInterf[m][p] = aux / N

        ##############################
        # Parâmetros exclusivos do Algorítmo Genético
        # Parâmetros definidos pela técnica de otimização Hyperas
        generations = 60                                # hyperopt[10 a 100]
        population_size = 90                            # hyperopt[10 a 200]
        crossover_probability = 0.7822617332190482      # hyperopt[0.1 a 1 ]
        mutation_probability = 0.8697451566145131       # hyperopt[0.1 a 1 ]
        parents = 16                                    # hyperopt[2  a 20 ]
        elitism = 2                                     # hyperopt[1  a  2 ]

        parent_selection_type = "sss"       # Estado Estável - EE
        crossover_type = "single_point"     # Ponto-simples  - PS
        mutation_type = "scramble"          # Embaralhamento - Em

        # Cria um indivíduo aleatoriamente
        def individual():
            NreNodeBs = random.randint(Interc, Ant)
            individual = np.zeros((M, P))
            for i in range(NreNodeBs):
                individual[random.randint(0, M - 1)][random.randint(0, P-1)] = 1
            return individual.reshape(M*P)

        def fitness(solution, solution_idx):
            sol = solution.reshape(M,P)

            # Maximização da cobertura
            cliNaoAtendidos = np.zeros((N))
            for n in range(N):
                cliNaoAtendidos[n] = Nij[nn[n][0]][nn[n][1]]
            for m in range(M):
                aux = 0 # Nr de clientes atendidos por eNodeB
                for n in range(N):
                    for p in range(P):
                        if aux < Usu and sol[m][p] == 1 and Cmnp[m][n][p] == 1 and cliNaoAtendidos[n] > 0:
                            cliAux = cliNaoAtendidos[n]
                            aux1 = aux
                            aux += cliNaoAtendidos[n]
                            cliNaoAtendidos[n] = 0
                            if aux > Usu:
                                cliNaoAtendidos[n] = cliAux
                                aux = aux1
            nrTotalClientes = 0
            nrClientesNaoAtendidos = 0
            for n in range(N):
                nrTotalClientes += Nij[nn[n][0]][nn[n][1]]
                nrClientesNaoAtendidos += cliNaoAtendidos[n]
            nrClientesAtendidos = nrTotalClientes - nrClientesNaoAtendidos

            # Minimização da interferência
            interf = 0
            for m in range(M):
                for p in range(P):
                    interf += grauInterf[m][p] * sol[m][p]

            # Minimização do número de eNodeBs instaladas
            nr_eNodeBs = 0
            for m in range(M):
                for p in range(P):
                    nr_eNodeBs += sol[m][p]

            # Restrição (6): Taxa de interconexão
            interconectado = True
            for m in range(M):
                for p in range(P):
                    grauInterc = 0
                    if sol[m][p] == 1:
                        for k in range(M):
                            for l in range(P):
                                grauInterc += sol[k][l] * A[m][k]
                        if grauInterc < (Interc+1):
                            interconectado = False

            # Restrição (5): Número máximo de eNodeBs instaladas
            nr_eNodeBs_isOK = True
            if nr_eNodeBs > Ant:
                nr_eNodeBs_isOK = False

            # Restrição da Maximização da Área de Cobertura
            nr_clientes_isOK = True
            if nrClientesAtendidos < consensoCobertura * nrTotalClientes:
                nr_clientes_isOK = False

            fit = -1/8 * interf
            if not interconectado: # Punição para o não atendimento à taxa de interconexão
                fit -= 10
            if not nr_eNodeBs_isOK: # Punição para o excesso de eNodeBs
                fit -= nr_eNodeBs
            if not nr_clientes_isOK: # Punição para o não atendimento do consenso de cobertura
                fit -= 10
            return fit

        def create_population():
            return [individual() for i in range(population_size)]

        def callback_generation(ga_instance):     #CallBack para acessar as informações após cada geração
            global lastBestfitness
            print("--------------------------------------------------------------")
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))  # [0] = best_solution, [1] = best_fitness, [2] = índice do best_solution
            print("Change     = {change}".format(change=ga_instance.best_solution()[1] - lastBestfitness))
            print("--------------------------------------------------------------")
            lastBestfitness = ga_instance.best_solution()[1]


        iteracoes = 30
        antes_global = 0
        lastBestfitness = 0

        # time, bFit, nr_eNodeBs, nr_cliAtendidos, areaCoberta, sobrePos2, sobrePos3, sobrePos4, sobrePos5, sobrePos6, sobrePos7, sobrePos8, sobrePos9, PRD
        result = np.zeros((iteracoes, 14))

        for it in range(iteracoes):
            print("--------------------------------------------------------------")
            print("------------------- Iteração {}/{} --------------------------".format(it+1,iteracoes))
            print("--------------------------------------------------------------")
            antes_global = t.time()
            ga_instance = pygad.GA(num_generations=generations,
                                   num_parents_mating=parents,
                                   fitness_func=fitness,
                                   initial_population=create_population(),
                                   gene_type=int,
                                   parent_selection_type=parent_selection_type,
                                   K_tournament=parents,
                                   keep_parents=elitism,
                                   crossover_type=crossover_type,
                                   crossover_probability=crossover_probability,
                                   mutation_type=mutation_type,
                                   mutation_probability=mutation_probability,
                                   on_generation=callback_generation)
            ga_instance.run()

            ###################################################
            # Avaliando a solução e armazenando suas métricas #
            ###################################################
            sol = ga_instance.best_solution()[0].reshape(M, P)

            result[it][0] = t.time() - antes_global                                             # time
            result[it][1] = ga_instance.best_solution()[1]                                      # bFit

            aux = 0
            for m in range(M):
                for p in range(P):
                    aux += sol[m][p]
            result[it][2] = aux                                                                 # nr_eNodeBs

            # nr_CliAtendidos
            cliNaoAtendidos = np.zeros((N))
            for n in range(N):
                cliNaoAtendidos[n] = Nij[nn[n][0]][nn[n][1]]

            for m in range(M):
                aux = 0  # Nr de clientes atendidos por eNodeB
                pAtiva = None
                quadriculasAtendidas = []
                for n in range(N):
                    for p in range(P):
                        if aux < Usu and sol[m][p] == 1 and Cmnp[m][n][p] == 1 and cliNaoAtendidos[n] > 0:
                            cliAux = cliNaoAtendidos[n]
                            aux1 = aux
                            aux += cliNaoAtendidos[n]
                            cliNaoAtendidos[n] = 0
                            quadriculasAtendidas.append(n)
                            pAtiva = p
                            if aux > Usu:
                                cliNaoAtendidos[n] = cliAux
                                aux = aux1
                if aux != 0:
                    print("Clientes atendidos pela eNodeB [{},{}] = {}".format(m, pAtiva, aux))

            nrTotalClientes = 0
            nrClientesNaoAtendidos = 0
            for n in range(N):
                nrTotalClientes += Nij[nn[n][0]][nn[n][1]]
                nrClientesNaoAtendidos += cliNaoAtendidos[n]

            result[it][3] = nrTotalClientes - nrClientesNaoAtendidos                            # nr_cliAtendidos

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
            pTx = [20, 25, 30, 35, 40]  # Potências de Tx avaliadas

            ##########      Parâmetros para o cálculo do Path Loss      ##########
            dBP = 0.22  # Break Point Distance
            fc = 700  # Frequency Carrier = 700MHz (frequência do 4G/LTE do CCOp Mv)
            hBS = 10  # Altura da eNodeB = 10m
            hUT = 1.5  # Altura dos UEs = 1.5m
            h = 5  # Altura média das construções = 5m (definido para o cenário RMa)
            W = 20  # Largura média das ruas = 20m (definido para o cenário RMa)
            #######################################################################
            eNB_pot = []
            for m in range(M):
                for p in range(P):
                    if sol[m][p] == 1:
                        eNB_pot.append((m, p))

            for eNB in eNB_pot:
                m = eNB[0]
                mi = mm[m][0]
                mj = mm[m][1]
                p = pTx[eNB[1]]
                for i in range(I):
                    for j in range(J):
                        dist = d.GeodesicDistance(Cij[mi][mj], Cij[i][j]).km
                        # Path Loss (PL) para RMa NLOS
                        if dist > 0.01:
                            PL = 161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h / hBS) ** 2) * math.log10(hBS) + (43.42 - 3.1 * math.log10(hBS)) * (math.log10(dist * 1000) - 3) + 20 * math.log10(fc / 1000) - (3.2 * (math.log10(11.75 * hUT)) ** 2 - 4.97)
                        else:
                            PL = 0
                        pRx = p + 13 - PL
                        if (pRx > -100):
                            Cobertura[i][j] += 1
            print(Cobertura)

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

            result[it][4] = quadCobertas * areaDaQuadricula                                     # areaCoberta

            result[it][5] = quadSobrepostas2 * areaDaQuadricula                                 # sobrePos2
            result[it][6] = quadSobrepostas3 * areaDaQuadricula                                 # sobrePos3
            result[it][7] = quadSobrepostas4 * areaDaQuadricula                                 # sobrePos4
            result[it][8] = quadSobrepostas5 * areaDaQuadricula                                 # sobrePos5
            result[it][9] = quadSobrepostas6 * areaDaQuadricula                                 # sobrePos6
            result[it][10] = quadSobrepostas7 * areaDaQuadricula                                # sobrePos7
            result[it][11] = quadSobrepostas8 * areaDaQuadricula                                # sobrePos8
            result[it][12] = quadSobrepostas9 * areaDaQuadricula                                # sobrePos9

            result[it][13] = 100 * (otimo - result[it][1]) / abs(otimo)                         # PRD

        print("###############          {}/{}           ################".format(Interc, consensoCobertura))

        result = np.asarray(result)
        np.savetxt('result/GA{}{}_i{}c{}.txt'.format(I,J,Interc,consensoCobertura), result)