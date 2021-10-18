import pygad
import numpy
import random
import time
import matplotlib.pyplot as plt
import parametros as p

def individual():
    vetorAux = [-1] * p.A
    for i in range(p.A):
        valorAux = random.randint(0, p.individual_size - 1)
        if valorAux not in vetorAux:
            vetorAux[i] = valorAux
    individual = [0] * p.individual_size
    for i in vetorAux:
        individual[i] = 1
    return individual

def fitness(solution, solution_idx):
    fitness = 0
    for i in range(p.M):
        aux = 0
        for j in range(p.N):
            aux += p.C[i][j] * solution[j]
        if aux > 0:
            fitness += 1
        if aux > 1:
            fitness -= 1
    return fitness

def create_population():
    return [individual() for i in range(p.population_size)]

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
    if best_fitness > p.objInExactAlgo[p.A]:
        y[i_global][ga_instance.generations_completed - 1] += (p.objInExactAlgo[p.A])/iteracoes
    else:
        y[i_global][ga_instance.generations_completed - 1] += (best_fitness)/iteracoes

    # Captura do Tempo e Geração para atingir 80%, 90% e 95% da solução ótima
    if t80[i_global][it_global] == 0 and p.obj80 <= ga_instance.best_solution()[1]:
        t80[i_global][it_global] = time.time() - antes_global
        g80[i_global][it_global] = ga_instance.generations_completed
    elif t90[i_global][it_global] == 0 and p.obj90 <= ga_instance.best_solution()[1]:
        t90[i_global][it_global] = time.time() - antes_global
        g90[i_global][it_global] = ga_instance.generations_completed
    elif t95[i_global][it_global] == 0 and p.obj95 <= ga_instance.best_solution()[1]:
        t95[i_global][it_global] = time.time() - antes_global
        g95[i_global][it_global] = ga_instance.generations_completed

########################################################################################
iteracoes = 200    ######################################################################
########################################################################################
lastBestfitness = 0
i_global = 0    # variável que controla a mudança dos fatores (métodos de seleção, métodos de crossover e de mutação)
it_global = 0
antes_global = 0
parent_selection_type = ['sss', 'rws', 'sus', 'rank', 'random', 'tournament']
#crossover_type = ['single_point', 'two_points', 'uniform', 'scattered']
#mutation_type = ['random', 'swap', 'inversion', 'scramble', 'adaptive']

# Variáveis utilizadas para impressão dos gráficos de convergência
y = numpy.zeros((len(parent_selection_type), p.generations))            # --> mudar o array para avaliar outro fator
x = numpy.arange(p.generations)

# Variáveis utilizadas para os boxplots e arquivo
yTime = numpy.zeros((iteracoes, len(parent_selection_type)))            # --> mudar o array para avaliar outro fator
yBFit = numpy.zeros((iteracoes, len(parent_selection_type)))            # --> mudar o array para avaliar outro fator

# Variáveis de velocidade para arquivo
t80 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator
t90 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator
t95 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator
g80 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator
g90 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator
g95 = numpy.zeros((len(parent_selection_type), iteracoes))              # --> mudar o array para avaliar outro fator

for i in range(len(parent_selection_type)):                             # --> mudar o array para avaliar outro fator
    print("----------------------------Método {x}/{y}-------------------------------".format(x = i+1, y= len(parent_selection_type)))
    i_global = i

    for it in range(iteracoes):
        print("----------------------------Iteração {x}/{y}------------------------------".format(x= it+1, y=iteracoes))
        it_global = it
        antes_global = time.time()

        ga_instance = pygad.GA(num_generations=p.generations,
                               num_parents_mating=p.parents,
                               fitness_func=fitness,
                               initial_population=create_population(),
                               num_genes=p.N,
                               parent_selection_type=parent_selection_type[i],      # --> mudar o array para avaliar outro fator
                               keep_parents=p.elitism,
                               crossover_type=p.crossover_type,                     # --> mudar o array para avaliar outro fator
                               crossover_probability=p.crossover_probability,
                               mutation_type=p.mutation_type,                       # --> mudar o array para avaliar outro fator
                               mutation_percent_genes=p.mutation_probability,
                               on_generation=callback_generation)
        ga_instance.run()

# As informações após todas gerações são coletadas aqui
        yTime[it][i_global] = time.time() - antes_global
        yBFit[it][i_global] = ga_instance.best_solution()[1]



###########################
#####    IMPRESSÃO    #####
###########################

#####     GRÁFICO DE CONVERGÊNCIA     #####
plt.plot(x, numpy.ones(len(x)) * p.objInExactAlgo[p.A], 'b--', label='E_ALLOCATOR')
plt.plot(x, y[0], label='M-ALLOCATOR-EE')
plt.plot(x, y[1], label='M-ALLOCATOR-R')
plt.plot(x, y[2], label='M-ALLOCATOR-E')
plt.plot(x, y[3], label='M-ALLOCATOR-Tr')
plt.plot(x, y[4], label='M-ALLOCATOR-A')
plt.plot(x, y[5], label='M-ALLOCATOR-To')
#plt.title('Convergência do AG com variação dos métodos de seleção')
#plt.ylabel('Valor de aptidão médio de {} iterações'.format(iteracoes))
#plt.xlabel('Geração')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



#####   BOXPLOTS   #####

# Melhor fitness após todas gerações
#labels = ['Estado estável', 'Roleta', 'Estocástico', 'Truncamento','Aleatório', 'Torneio']
labels = ['EE', 'R', 'E', 'Tr','A', 'To']
plt.boxplot(yBFit, labels=labels)
#plt.suptitle('Distribuição dos melhores valores de aptidão alcançados')
#plt.title('por método de seleção')
#plt.ylabel('Melhor valor de aptidão')
#plt.xlabel('Resultados de {} iterações'.format(iteracoes))
plt.show()

# Tempo gasto no processamento
#labels = ['Estado estável', 'Roleta', 'Estocástico', 'Truncamento','Aleatório', 'Torneio']
labels = ['EE', 'R', 'E', 'Tr','A', 'To']
plt.boxplot(yTime, labels=labels)
#plt.title('Distribuição dos tempos de processamento por método de seleção')
#plt.ylabel('Tempo de processamento ')
#plt.xlabel('Resultados de {} iterações'.format(iteracoes))
plt.show()



#####   Salva em ARQUIVO para outras análises   #####
t = numpy.asarray(yTime)
b = numpy.asarray(yBFit)
t80 = numpy.asarray(t80)
t90 = numpy.asarray(t90)
t95 = numpy.asarray(t95)
g80 = numpy.asarray(g80)
g90 = numpy.asarray(g90)
g95 = numpy.asarray(g95)

numpy.savetxt('result_Selecao_time.txt', t, fmt="%f", delimiter=",")
numpy.savetxt('result_Selecao_bFit.txt', b, fmt="%d", delimiter=",")
numpy.savetxt('result_Selecao_t80.txt', t80, fmt="%f", delimiter=",")
numpy.savetxt('result_Selecao_t90.txt', t90, fmt="%f", delimiter=",")
numpy.savetxt('result_Selecao_t95.txt', t95, fmt="%f", delimiter=",")
numpy.savetxt('result_Selecao_g80.txt', g80, fmt="%d", delimiter=",")
numpy.savetxt('result_Selecao_g90.txt', g90, fmt="%d", delimiter=",")
numpy.savetxt('result_Selecao_g95.txt', g95, fmt="%d", delimiter=",")