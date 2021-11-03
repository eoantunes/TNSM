import pygad
import numpy as np
import pandas
import os
import random

from hyperopt import fmin, tpe, STATUS_OK, Trials, hp

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
Interc = 3 # Nr mínimo de nós interconectados (uma eNodeB precisa estar conectada a mais Interc eNodeBs)(0, 1, 2, 3)
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
#generations = 80                                # hyperopt[10 a 100]
#population_size = 180                           # hyperopt[20 a 200]
#crossover_probability = 0.9253311194295362      # hyperopt[0.1 a 1 ]
#mutation_probability = 0.7169036384337947       # hyperopt[10 a 100]
#parents = 18                                    # hyperopt[2  a 20 ]       # Número de pais a serem selecionados
#elitism = 2                                     # hyperopt[1  a  2 ]


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


def create_population(popSize):
    return [individual() for i in range(popSize)]

def callback_generation(ga_instance):     #CallBack para acessar as informações após cada geração
    global lastBestfitness
    print("--------------------------------------------------------------")
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))  # [0] = best_solution, [1] = best_fitness, [2] = índice do best_solution
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - lastBestfitness))
    print("--------------------------------------------------------------")
    lastBestfitness = ga_instance.best_solution()[1]


space = {
    'generations': hp.choice('generations', list(range(10, 110, 10))),
    'population_size': hp.choice('population_size', list(range(20, 210, 10))),
    'crossover-probability': hp.uniform('crossover-probability', 0.1, 1),
    'mutation-probability':  hp.uniform('mutation-probability' , 0.1, 1),
    'parents': hp.choice('parents', list(range(2, 21))),
    'elitism': hp.choice('elitism', list(range(1, 3)))
}


def objective(params):
    ga_instance = pygad.GA(num_generations=params['generations'],
                           num_parents_mating=params['parents'],
                           fitness_func=fitness,
                           initial_population=create_population(params['population_size']),
                           gene_type=int,
                           parent_selection_type=parent_selection_type,
                           #K_tournament=parents,
                           keep_parents=params['elitism'],
                           crossover_type=crossover_type,
                           crossover_probability=params['crossover-probability'],
                           mutation_type=mutation_type,
                           mutation_probability=params['mutation-probability'],
                           on_generation=callback_generation)
    ga_instance.run()

    _, solution_fitness, _ = ga_instance.best_solution()

    return {'loss': -solution_fitness, 'status': STATUS_OK}

lastBestfitness = 0
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)
df = pandas.DataFrame()

trial_dict = {}
for t in trials.trials:
    trial_dict.update(t['misc']['vals'])
    trial_dict.update(t['result'])
    df = df.append(trial_dict, ignore_index=True)

print(best)
print(trials.best_trial)

outname = 'hyperopt_GA.csv'
outdir = 'resultHyperOpt'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

df.to_csv(fullname, mode='a')

best_df = pandas.DataFrame()
best_df = best_df.append(trials.best_trial, ignore_index=True)

outname = 'best_trial.csv'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

best_df.to_csv(fullname, mode='a')