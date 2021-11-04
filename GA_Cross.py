import pygad
import numpy as np
import random
import time
import matplotlib.pyplot as plt

###   Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
I,J = 12,15

Mij = np.loadtxt('matriz/M{}{}.txt'.format(I,J))
Nij = np.loadtxt('matriz/N{}{}.txt'.format(I,J))
loaded_Smnp = np.loadtxt('matriz/Smnp{}{}.txt'.format(I,J))
Smnp = loaded_Smnp.reshape(loaded_Smnp.shape[0], loaded_Smnp.shape[1] // 5, 5)
loaded_Cmnp = np.loadtxt('matriz/Cmnp{}{}.txt'.format(I,J))
Cmnp = loaded_Cmnp.reshape(loaded_Cmnp.shape[0], loaded_Cmnp.shape[1] // 5, 5)
A = np.loadtxt('matriz/A{}{}.txt'.format(I,J))

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
Interc = 1 # Nr mínimo de nós interconectados (uma eNodeB precisa estar conectada a mais Interc eNodeBs)(0, 1, 2, 3)
consensoCobertura = 0.95 # Porcentagem do número de clientes que devem ser atendidos (0.8, 0.9, 0.95, 0.98, 1)

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
parents = 16                                    # hyperopt[2  a 20 ]       # Número de pais a serem selecionados
elitism = 2                                     # hyperopt[1  a  2 ]


##########   FATORES DA AVALIAÇÃO   ##########
# Métodos de Seleção (parent_selection_type):
"""
sss (for steady-state selection),           <- (estado estável)
rws (for roulette wheel selection),         <- (roleta)
sus (for stochastic universal selection),   <- (estocástico) 
rank (for rank selection),                  <- (truncamento)
random (for random selection),              <- (aleatório) 
tournament (for tournament selection)       <- (torneio)
"""
parent_selection_type = "sss"


# Métodos de Reprodução (crossover_type):
"""
single_point (for single-point crossover),  <- (ponto simples)
two_points (for two points crossover),      <- (dois pontos)
uniform (for uniform crossover),            <- (uniforme)
scattered (for scattered crossover).        <- (espalhamento)
"""
crossover_type = "two_points"


# Métodos de Mutação (mutation_type):
"""
random (for random mutation),               <- (aleatório)
swap (for swap mutation),                   <- (troca)
inversion (for inversion mutation),         <- (inversão)
scramble (for scramble mutation),           <- (embaralhamento)
adaptive (for adaptive mutation).           <- (adaptativo)
"""
mutation_type = "scramble"

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

    # Captura do BestFitness para os gráficos de convergência
    best_fitness = ga_instance.best_solution()[1]
    y[i_global][ga_instance.generations_completed - 1] += (best_fitness)/iteracoes

########################################################################################
iteracoes = 100    ######################################################################
########################################################################################
lastBestfitness = 0
i_global = 0    # variável que controla a mudança dos fatores (métodos de seleção, métodos de crossover e de mutação)
it_global = 0
antes_global = 0
#parent_selection_type = ['sss', 'rws', 'sus', 'rank', 'random', 'tournament']
crossover_type = ['single_point', 'two_points', 'uniform', 'scattered']
#mutation_type = ['random', 'swap', 'inversion', 'scramble', 'adaptive']

# Variáveis utilizadas para impressão dos gráficos de convergência
y = np.zeros((len(crossover_type), generations))            # --> mudar o array para avaliar outro fator
x = np.arange(generations)

# Variáveis utilizadas para os boxplots e arquivo
yTime = np.zeros((iteracoes, len(crossover_type)))            # --> mudar o array para avaliar outro fator
yBFit = np.zeros((iteracoes, len(crossover_type)))            # --> mudar o array para avaliar outro fator

for i in range(len(crossover_type)):                             # --> mudar o array para avaliar outro fator
    print("----------------------------Método {x}/{y}-------------------------------".format(x = i+1, y= len(crossover_type))) # --> mudar o array para avaliar outro fator
    i_global = i

    for it in range(iteracoes):
        print("----------------------------Iteração {x}/{y}------------------------------".format(x= it+1, y=iteracoes))
        it_global = it
        antes_global = time.time()

        ga_instance = pygad.GA(num_generations=generations,
                               num_parents_mating=parents,
                               fitness_func=fitness,
                               initial_population=create_population(),
                               gene_type=int,
                               parent_selection_type=parent_selection_type,         # --> mudar o array para avaliar outro fator
                               K_tournament=parents,
                               keep_parents=elitism,
                               crossover_type=crossover_type[i],                    # --> mudar o array para avaliar outro fator
                               crossover_probability=crossover_probability,
                               mutation_type=mutation_type,                         # --> mudar o array para avaliar outro fator
                               mutation_probability=mutation_probability,
                               on_generation=callback_generation)
        ga_instance.run()

# As informações após todas gerações são coletadas aqui
        yTime[it][i_global] = time.time() - antes_global
        yBFit[it][i_global] = ga_instance.best_solution()[1]
        ###   TESTES   ###
        sol = ga_instance.best_solution()[0].reshape(M,P)





        # Maximização da cobertura
        cliNaoAtendidos = np.zeros((N))
        for n in range(N):
            cliNaoAtendidos[n] = Nij[nn[n][0]][nn[n][1]]
        #print(cliNaoAtendidos)

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
                #print("Quadrículas antendidas: {}".format(quadriculasAtendidas))
                #print("Clientes NÃO atendidos: {}".format(cliNaoAtendidos))

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




        print("Número de clientes atendidos = {}".format(nrClientesAtendidos))
        print()
        print("eNodeBs instaladas / grau de interferência: ")
        for m in range(M):
            for p in range(P):
                if sol[m][p] == 1:
                    print("Y[{}][{}] / {}".format(m,p,grauInterf[m][p]))
        print()
        print("Número de eNodeBs instalados = {}".format(nr_eNodeBs))
        print("Interferência = {}".format(interf))
        print("Bestfitness = {}".format(ga_instance.best_solution()[1]))


###########################
#####    IMPRESSÃO    #####
###########################

#####     GRÁFICO DE CONVERGÊNCIA     #####
plt.plot(x, np.ones(len(x)) * -0.11148648648648649, 'b--', label='E_ALLOCATOR')
plt.plot(x, y[0], label='M-ALLOCATOR-PS')
plt.plot(x, y[1], label='M-ALLOCATOR-DP')
plt.plot(x, y[2], label='M-ALLOCATOR-U')
plt.plot(x, y[3], label='M-ALLOCATOR-Es')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



#####   BOXPLOTS   #####

# Melhor fitness após todas gerações
labels = ['PS', 'DP', 'U', 'Es']
plt.boxplot(yBFit, labels=labels)
plt.grid(axis='y')
plt.show()

# Tempo gasto no processamento
labels = ['PS', 'DP', 'U', 'Es']
plt.boxplot(yTime, labels=labels)
plt.grid(axis='y')
plt.show()



#####   Salva em ARQUIVO para outras análises   #####
t = np.asarray(yTime)
b = np.asarray(yBFit)
np.savetxt('result_Cross_time.txt', t, fmt="%f")
np.savetxt('result_Cross_bFit.txt', b, fmt="%d")