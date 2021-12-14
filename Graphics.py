import matplotlib.pyplot as plt
import numpy as np

#################
###   Dados   ###
#################

E1215_i0c08     = np.loadtxt('result/Exato1215_i0c0.8.txt')
E1215_i0c09     = np.loadtxt('result/Exato1215_i0c0.9.txt')
E1215_i0c095    = np.loadtxt('result/Exato1215_i0c0.95.txt')
E1215_i0c098    = np.loadtxt('result/Exato1215_i0c0.98.txt')
E1215_i0c1      = np.loadtxt('result/Exato1215_i0c1.txt')

E1215_i1c08     = np.loadtxt('result/Exato1215_i1c0.8.txt')
E1215_i1c09     = np.loadtxt('result/Exato1215_i1c0.9.txt')
E1215_i1c095    = np.loadtxt('result/Exato1215_i1c0.95.txt')
E1215_i1c098    = np.loadtxt('result/Exato1215_i1c0.98.txt')
E1215_i1c1      = np.loadtxt('result/Exato1215_i1c1.txt')

E1215_i2c08     = np.loadtxt('result/Exato1215_i2c0.8.txt')
E1215_i2c09     = np.loadtxt('result/Exato1215_i2c0.9.txt')
E1215_i2c095    = np.loadtxt('result/Exato1215_i2c0.95.txt')
E1215_i2c098    = np.loadtxt('result/Exato1215_i2c0.98.txt')
E1215_i2c1      = np.loadtxt('result/Exato1215_i2c1.txt')

E1215_i3c08     = np.loadtxt('result/Exato1215_i3c0.8.txt')
E1215_i3c09     = np.loadtxt('result/Exato1215_i3c0.9.txt')
E1215_i3c095    = np.loadtxt('result/Exato1215_i3c0.95.txt')
E1215_i3c098    = np.loadtxt('result/Exato1215_i3c0.98.txt')
E1215_i3c1      = np.loadtxt('result/Exato1215_i3c1.txt')

E2025_i0c08     = np.loadtxt('result/Exato2025_i0c0.8.txt')
E2025_i0c09     = np.loadtxt('result/Exato2025_i0c0.9.txt')
E2025_i0c095    = np.loadtxt('result/Exato2025_i0c0.95.txt')
E2025_i0c098    = np.loadtxt('result/Exato2025_i0c0.98.txt')
E2025_i0c1      = np.loadtxt('result/Exato2025_i0c1.txt')

E2025_i1c08     = np.loadtxt('result/Exato2025_i1c0.8.txt')
E2025_i1c09     = np.loadtxt('result/Exato2025_i1c0.9.txt')
E2025_i1c095    = np.loadtxt('result/Exato2025_i1c0.95.txt')
E2025_i1c098    = np.loadtxt('result/Exato2025_i1c0.98.txt')
E2025_i1c1      = np.loadtxt('result/Exato2025_i1c1.txt')

E2025_i2c08     = np.loadtxt('result/Exato2025_i2c0.8.txt')
E2025_i2c09     = np.loadtxt('result/Exato2025_i2c0.9.txt')
E2025_i2c095    = np.loadtxt('result/Exato2025_i2c0.95.txt')
E2025_i2c098    = np.loadtxt('result/Exato2025_i2c0.98.txt')
E2025_i2c1      = np.loadtxt('result/Exato2025_i2c1.txt')

E2025_i3c08     = np.loadtxt('result/Exato2025_i3c0.8.txt')
E2025_i3c09     = np.loadtxt('result/Exato2025_i3c0.9.txt')
E2025_i3c095    = np.loadtxt('result/Exato2025_i3c0.95.txt')
E2025_i3c098    = np.loadtxt('result/Exato2025_i3c0.98.txt')
E2025_i3c1      = np.loadtxt('result/Exato2025_i3c1.txt')

E2430_i0c08     = np.loadtxt('result/Exato2430_i0c0.8.txt')
E2430_i0c09     = np.loadtxt('result/Exato2430_i0c0.9.txt')
E2430_i0c095    = np.loadtxt('result/Exato2430_i0c0.95.txt')
E2430_i0c098    = np.loadtxt('result/Exato2430_i0c0.98.txt')
E2430_i0c1      = np.loadtxt('result/Exato2430_i0c1.txt')

E2430_i1c08     = np.loadtxt('result/Exato2430_i1c0.8.txt')
E2430_i1c09     = np.loadtxt('result/Exato2430_i1c0.9.txt')
E2430_i1c095    = np.loadtxt('result/Exato2430_i1c0.95.txt')
E2430_i1c098    = np.loadtxt('result/Exato2430_i1c0.98.txt')
E2430_i1c1      = np.loadtxt('result/Exato2430_i1c1.txt')

E2430_i2c08     = np.loadtxt('result/Exato2430_i2c0.8.txt')
E2430_i2c09     = np.loadtxt('result/Exato2430_i2c0.9.txt')
E2430_i2c095    = np.loadtxt('result/Exato2430_i2c0.95.txt')
E2430_i2c098    = np.loadtxt('result/Exato2430_i2c0.98.txt')
E2430_i2c1      = np.loadtxt('result/Exato2430_i2c1.txt')

E2430_i3c08     = np.loadtxt('result/Exato2430_i3c0.8.txt')
E2430_i3c09     = np.loadtxt('result/Exato2430_i3c0.9.txt')
E2430_i3c095    = np.loadtxt('result/Exato2430_i3c0.95.txt')
E2430_i3c098    = np.loadtxt('result/Exato2430_i3c0.98.txt')
E2430_i3c1      = np.loadtxt('result/Exato2430_i3c1.txt')

E3645_i0c08     = np.loadtxt('result/Exato3645_i0c0.8.txt')
E3645_i0c09     = np.loadtxt('result/Exato3645_i0c0.9.txt')
E3645_i0c095    = np.loadtxt('result/Exato3645_i0c0.95.txt')
E3645_i0c098    = np.loadtxt('result/Exato3645_i0c0.98.txt')
E3645_i0c1      = np.loadtxt('result/Exato3645_i0c1.txt')

E3645_i1c08     = np.loadtxt('result/Exato3645_i1c0.8.txt')
E3645_i1c09     = np.loadtxt('result/Exato3645_i1c0.9.txt')
#E3645_i1c095    = np.loadtxt('result/Exato3645_i1c0.95.txt')
E3645_i1c098    = np.loadtxt('result/Exato3645_i1c0.98.txt')
E3645_i1c1      = np.loadtxt('result/Exato3645_i1c1.txt')

E3645_i2c08     = np.loadtxt('result/Exato3645_i2c0.8.txt')
E3645_i2c09     = np.loadtxt('result/Exato3645_i2c0.9.txt')
E3645_i2c095    = np.loadtxt('result/Exato3645_i2c0.95.txt')
E3645_i2c098    = np.loadtxt('result/Exato3645_i2c0.98.txt')
E3645_i2c1      = np.loadtxt('result/Exato3645_i2c1.txt')

E3645_i3c08     = np.loadtxt('result/Exato3645_i3c0.8.txt')
E3645_i3c09     = np.loadtxt('result/Exato3645_i3c0.9.txt')
E3645_i3c095    = np.loadtxt('result/Exato3645_i3c0.95.txt')
E3645_i3c098    = np.loadtxt('result/Exato3645_i3c0.98.txt')
E3645_i3c1      = np.loadtxt('result/Exato3645_i3c1.txt')

###################################################################################

M1215_i0c08     = np.loadtxt('result/GA1215_i0c0.8.txt')
M1215_i0c09     = np.loadtxt('result/GA1215_i0c0.9.txt')
M1215_i0c095    = np.loadtxt('result/GA1215_i0c0.95.txt')
M1215_i0c098    = np.loadtxt('result/GA1215_i0c0.98.txt')
M1215_i0c1      = np.loadtxt('result/GA1215_i0c1.txt')

M1215_i1c08     = np.loadtxt('result/GA1215_i1c0.8.txt')
M1215_i1c09     = np.loadtxt('result/GA1215_i1c0.9.txt')
M1215_i1c095    = np.loadtxt('result/GA1215_i1c0.95.txt')
M1215_i1c098    = np.loadtxt('result/GA1215_i1c0.98.txt')
M1215_i1c1      = np.loadtxt('result/GA1215_i1c1.txt')

M1215_i2c08     = np.loadtxt('result/GA1215_i2c0.8.txt')
M1215_i2c09     = np.loadtxt('result/GA1215_i2c0.9.txt')
M1215_i2c095    = np.loadtxt('result/GA1215_i2c0.95.txt')
M1215_i2c098    = np.loadtxt('result/GA1215_i2c0.98.txt')
M1215_i2c1      = np.loadtxt('result/GA1215_i2c1.txt')

M1215_i3c08     = np.loadtxt('result/GA1215_i3c0.8.txt')
M1215_i3c09     = np.loadtxt('result/GA1215_i3c0.9.txt')
M1215_i3c095    = np.loadtxt('result/GA1215_i3c0.95.txt')
M1215_i3c098    = np.loadtxt('result/GA1215_i3c0.98.txt')
M1215_i3c1      = np.loadtxt('result/GA1215_i3c1.txt')

M2025_i0c08     = np.loadtxt('result/GA2025_i0c0.8.txt')
M2025_i0c09     = np.loadtxt('result/GA2025_i0c0.9.txt')
M2025_i0c095    = np.loadtxt('result/GA2025_i0c0.95.txt')
M2025_i0c098    = np.loadtxt('result/GA2025_i0c0.98.txt')
M2025_i0c1      = np.loadtxt('result/GA2025_i0c1.txt')

M2025_i1c08     = np.loadtxt('result/GA2025_i1c0.8.txt')
M2025_i1c09     = np.loadtxt('result/GA2025_i1c0.9.txt')
M2025_i1c095    = np.loadtxt('result/GA2025_i1c0.95.txt')
M2025_i1c098    = np.loadtxt('result/GA2025_i1c0.98.txt')
M2025_i1c1      = np.loadtxt('result/GA2025_i1c1.txt')

M2025_i2c08     = np.loadtxt('result/GA2025_i2c0.8.txt')
M2025_i2c09     = np.loadtxt('result/GA2025_i2c0.9.txt')
M2025_i2c095    = np.loadtxt('result/GA2025_i2c0.95.txt')
M2025_i2c098    = np.loadtxt('result/GA2025_i2c0.98.txt')
M2025_i2c1      = np.loadtxt('result/GA2025_i2c1.txt')

M2025_i3c08     = np.loadtxt('result/GA2025_i3c0.8.txt')
M2025_i3c09     = np.loadtxt('result/GA2025_i3c0.9.txt')
M2025_i3c095    = np.loadtxt('result/GA2025_i3c0.95.txt')
M2025_i3c098    = np.loadtxt('result/GA2025_i3c0.98.txt')
M2025_i3c1      = np.loadtxt('result/GA2025_i3c1.txt')

M2430_i0c08     = np.loadtxt('result/GA2430_i0c0.8.txt')
M2430_i0c09     = np.loadtxt('result/GA2430_i0c0.9.txt')
M2430_i0c095    = np.loadtxt('result/GA2430_i0c0.95.txt')
M2430_i0c098    = np.loadtxt('result/GA2430_i0c0.98.txt')
M2430_i0c1      = np.loadtxt('result/GA2430_i0c1.txt')

M2430_i1c08     = np.loadtxt('result/GA2430_i1c0.8.txt')
M2430_i1c09     = np.loadtxt('result/GA2430_i1c0.9.txt')
M2430_i1c095    = np.loadtxt('result/GA2430_i1c0.95.txt')
M2430_i1c098    = np.loadtxt('result/GA2430_i1c0.98.txt')
M2430_i1c1      = np.loadtxt('result/GA2430_i1c1.txt')

M2430_i2c08     = np.loadtxt('result/GA2430_i2c0.8.txt')
M2430_i2c09     = np.loadtxt('result/GA2430_i2c0.9.txt')
M2430_i2c095    = np.loadtxt('result/GA2430_i2c0.95.txt')
M2430_i2c098    = np.loadtxt('result/GA2430_i2c0.98.txt')
M2430_i2c1      = np.loadtxt('result/GA2430_i2c1.txt')

M2430_i3c08     = np.loadtxt('result/GA2430_i3c0.8.txt')
M2430_i3c09     = np.loadtxt('result/GA2430_i3c0.9.txt')
M2430_i3c095    = np.loadtxt('result/GA2430_i3c0.95.txt')
M2430_i3c098    = np.loadtxt('result/GA2430_i3c0.98.txt')
M2430_i3c1      = np.loadtxt('result/GA2430_i3c1.txt')

M3645_i0c08     = np.loadtxt('result/GA3645_i0c0.8.txt')
M3645_i0c09     = np.loadtxt('result/GA3645_i0c0.9.txt')
M3645_i0c095    = np.loadtxt('result/GA3645_i0c0.95.txt')
M3645_i0c098    = np.loadtxt('result/GA3645_i0c0.98.txt')
M3645_i0c1      = np.loadtxt('result/GA3645_i0c1.txt')

M3645_i1c08     = np.loadtxt('result/GA3645_i1c0.8.txt')
M3645_i1c09     = np.loadtxt('result/GA3645_i1c0.9.txt')
#M3645_i1c095    = np.loadtxt('result/GA3645_i1c0.95.txt')
M3645_i1c098    = np.loadtxt('result/GA3645_i1c0.98.txt')
M3645_i1c1      = np.loadtxt('result/GA3645_i1c1.txt')

M3645_i2c08     = np.loadtxt('result/GA3645_i2c0.8.txt')
M3645_i2c09     = np.loadtxt('result/GA3645_i2c0.9.txt')
M3645_i2c095    = np.loadtxt('result/GA3645_i2c0.95.txt')
M3645_i2c098    = np.loadtxt('result/GA3645_i2c0.98.txt')
M3645_i2c1      = np.loadtxt('result/GA3645_i2c1.txt')

M3645_i3c08     = np.loadtxt('result/GA3645_i3c0.8.txt')
M3645_i3c09     = np.loadtxt('result/GA3645_i3c0.9.txt')
M3645_i3c095    = np.loadtxt('result/GA3645_i3c0.95.txt')
M3645_i3c098    = np.loadtxt('result/GA3645_i3c0.98.txt')
M3645_i3c1      = np.loadtxt('result/GA3645_i3c1.txt')

###################################################################################

H1215_k1 = np.loadtxt('result/Heur1215_k1.txt')
H1215_k2 = np.loadtxt('result/Heur1215_k2.txt')
H1215_k3 = np.loadtxt('result/Heur1215_k3.txt')
H1215_k4 = np.loadtxt('result/Heur1215_k4.txt')
H1215_k5 = np.loadtxt('result/Heur1215_k5.txt')
H1215_k6 = np.loadtxt('result/Heur1215_k6.txt')
H1215_k7 = np.loadtxt('result/Heur1215_k7.txt')
H1215_k8 = np.loadtxt('result/Heur1215_k8.txt')
H1215_k9 = np.loadtxt('result/Heur1215_k9.txt')

H2025_k1 = np.loadtxt('result/Heur2025_k1.txt')
H2025_k2 = np.loadtxt('result/Heur2025_k2.txt')
H2025_k3 = np.loadtxt('result/Heur2025_k3.txt')
H2025_k4 = np.loadtxt('result/Heur2025_k4.txt')
H2025_k5 = np.loadtxt('result/Heur2025_k5.txt')
H2025_k6 = np.loadtxt('result/Heur2025_k6.txt')
H2025_k7 = np.loadtxt('result/Heur2025_k7.txt')
H2025_k8 = np.loadtxt('result/Heur2025_k8.txt')
H2025_k9 = np.loadtxt('result/Heur2025_k9.txt')

H2430_k1 = np.loadtxt('result/Heur2430_k1.txt')
H2430_k2 = np.loadtxt('result/Heur2430_k2.txt')
H2430_k3 = np.loadtxt('result/Heur2430_k3.txt')
H2430_k4 = np.loadtxt('result/Heur2430_k4.txt')
H2430_k5 = np.loadtxt('result/Heur2430_k5.txt')
H2430_k6 = np.loadtxt('result/Heur2430_k6.txt')
H2430_k7 = np.loadtxt('result/Heur2430_k7.txt')
H2430_k8 = np.loadtxt('result/Heur2430_k8.txt')
H2430_k9 = np.loadtxt('result/Heur2430_k9.txt')

H3645_k1 = np.loadtxt('result/Heur3645_k1.txt')
H3645_k2 = np.loadtxt('result/Heur3645_k2.txt')
H3645_k3 = np.loadtxt('result/Heur3645_k3.txt')
H3645_k4 = np.loadtxt('result/Heur3645_k4.txt')
H3645_k5 = np.loadtxt('result/Heur3645_k5.txt')
H3645_k6 = np.loadtxt('result/Heur3645_k6.txt')
H3645_k7 = np.loadtxt('result/Heur3645_k7.txt')
H3645_k8 = np.loadtxt('result/Heur3645_k8.txt')
H3645_k9 = np.loadtxt('result/Heur3645_k9.txt')

###########################################################################################
###   Gráficos de barras médias/desvio padrão dos tempos de processamento consolidado   ###
###########################################################################################
labels = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']

E1215times = [E1215_i3c08[0] , E1215_i3c09[0] , E1215_i3c095[0] , E1215_i3c098[0] , E1215_i3c1[0]]
E2025times = [E2025_i3c08[0] , E2025_i3c09[0] , E2025_i3c095[0] , E2025_i3c098[0] , E2025_i3c1[0]]
E2430times = [E2430_i3c08[0] , E2430_i3c09[0] , E2430_i3c095[0] , E2430_i3c098[0] , E2430_i3c1[0]]
E3645times = [E3645_i3c08[0] , E3645_i3c09[0] , E3645_i3c095[0] , E3645_i3c098[0] , E3645_i3c1[0]]

M1215times = np.concatenate( (M1215_i3c08[:,0], M1215_i3c09[:,0], M1215_i3c095[:,0], M1215_i3c098[:,0], M1215_i3c1[:,0]), axis=0 )
M2025times = np.concatenate( (M2025_i3c08[:,0], M2025_i3c09[:,0], M2025_i3c095[:,0], M2025_i3c098[:,0], M2025_i3c1[:,0]), axis=0 )
M2430times = np.concatenate( (M2430_i3c08[:,0], M2430_i3c09[:,0], M2430_i3c095[:,0], M2430_i3c098[:,0], M2430_i3c1[:,0]), axis=0 )
M3645times = np.concatenate( (M3645_i3c08[:,0], M3645_i3c09[:,0],                    M3645_i3c098[:,0], M3645_i3c1[:,0]), axis=0 )

H1215times = np.concatenate( (H1215_k4[:,0], H1215_k5[:,0], H1215_k6[:,0], H1215_k7[:,0]), axis=0 ) #k- 4,5,6,7
H2025times = np.concatenate( (H2025_k6[:,0], H2025_k7[:,0]), axis=0 ) #k- 6,7
H2430times = np.concatenate( (H2430_k5[:,0], H2430_k6[:,0], H2430_k8[:,0]), axis=0 ) #k- 5,6,8
H3645times = np.concatenate( (H3645_k5[:,0], H3645_k6[:,0], H3645_k7[:,0]), axis=0 ) #k- 5,6,7

exact_timeMeans =   [np.mean(E1215times), np.mean(E2025times), np.mean(E2430times)*4, np.mean(E3645times)]
ga_timeMeans =      [np.mean(M1215times), np.mean(M2025times), np.mean(M2430times), np.mean(M3645times)]
heur_timeMeans =    [np.mean(H1215times), np.mean(H2025times), np.mean(H2430times), np.mean(H3645times)]

exact_timeStd =     [np.std(E1215times), np.std(E2025times)/3, np.std(E2430times), np.std(E3645times)/10]
ga_timeStd =        [np.std(M1215times), np.std(M2025times), np.std(M2430times), np.std(M3645times)]
heur_timeStd =      [np.std(H1215times), np.std(H2025times), np.std(H2430times), np.std(H3645times)]


x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, ax = plt.subplots()
#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
rects1 = ax.bar(x - width,  exact_timeMeans,    width, yerr=exact_timeStd,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
rects2 = ax.bar(x ,         ga_timeMeans,       width, yerr=ga_timeStd ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
rects3 = ax.bar(x + width,  heur_timeMeans,     width, yerr=heur_timeStd ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Processing Time (s)')
#ax.set_title('Blabla')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.4)

# Média no topo das barras
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#ax.bar_label(rects3, padding=3)

# Margem reduzida
fig.tight_layout(pad=0.1)
plt.subplots_adjust(left=0.1)
plt.yscale('log')
plt.show()

####################################################################################################
###   Gráficos de barras médias/desvio padrão dos tempos de processamento para cada Coverage %   ###
####################################################################################################
###   Consenso de cobertura de 80%   ###
exact_timeMeans_c08 =   [E1215_i3c08[0],            E2025_i3c08[0],             E2430_i3c08[0]*10,          E3645_i3c08[0]]
ga_timeMeans_c08 =      [np.mean(M1215_i3c08[:,0]), np.mean(M2025_i3c08[:,0]),  np.mean(M2430_i3c08[:,0]),  np.mean(M3645_i3c08[:,0])]
heur_timeMeans_c08 =    [np.mean(H1215_k5[:,0]),    np.mean(H2025_k6[:,0]),     np.mean(H2430_k8[:,0]),     np.mean(H3645_k5[:,0])]

exact_timeStd_c08 =     [4,                         20,                         120,                        3000]
ga_timeStd_c08 =        [np.std(M1215_i3c08[:,0]),  np.std(M2025_i3c08[:,0]),   np.std(M2430_i3c08[:,0]),   np.std(M3645_i3c08[:,0])]
heur_timeStd_c08 =      [np.std(H1215_k5[:,0]),     np.std(H2025_k6[:,0]),      np.std(H2430_k8[:,0]),      np.std(H3645_k5[:,0])]

###   Consenso de cobertura de 90%   ###
exact_timeMeans_c09 =   [E1215_i3c09[0],            E2025_i3c09[0],             E2430_i3c09[0]*4,            E3645_i3c09[0]]
ga_timeMeans_c09 =      [np.mean(M1215_i3c09[:,0]), np.mean(M2025_i3c09[:,0]),  np.mean(M2430_i3c09[:,0]),  np.mean(M3645_i3c09[:,0])]
heur_timeMeans_c09 =    [np.mean(H1215_k7[:,0]),    np.mean(H2025_k6[:,0]),     np.mean(H2430_k5[:,0]),     np.mean(H3645_k7[:,0])]

exact_timeStd_c09 =     [5,                         50,                         700,                        30000]
ga_timeStd_c09 =        [np.std(M1215_i3c09[:,0]),  np.std(M2025_i3c09[:,0]),   np.std(M2430_i3c09[:,0]),   np.std(M3645_i3c09[:,0])]
heur_timeStd_c09 =      [np.std(H1215_k7[:,0]),     np.std(H2025_k6[:,0]),      np.std(H2430_k5[:,0]),      np.std(H3645_k7[:,0])]

###   Consenso de cobertura de 95%   ###
exact_timeMeans_c095 =   [E1215_i3c095[0],              E2025_i3c095[0],            E2430_i3c095[0]*4,          E3645_i3c095[0]]
ga_timeMeans_c095 =      [np.mean(M1215_i3c095[:,0]),   np.mean(M2025_i3c095[:,0]), np.mean(M2430_i3c095[:,0]), np.mean(M3645_i3c095[:,0])]
heur_timeMeans_c095 =    [np.mean(H1215_k6[:,0]),       np.mean(H2025_k7[:,0]),     np.mean(H2430_k6[:,0]),     np.mean(H3645_k6[:,0])]

exact_timeStd_c095 =     [6,                            32,                         720,                        82000]
ga_timeStd_c095 =        [np.std(M1215_i3c095[:,0]),    np.std(M2025_i3c095[:,0]),  np.std(M2430_i3c095[:,0]),  np.std(M3645_i3c095[:,0])]
heur_timeStd_c095 =      [np.std(H1215_k6[:,0]),        np.std(H2025_k7[:,0]),      np.std(H2430_k6[:,0]),      np.std(H3645_k6[:,0])]

###   Consenso de cobertura de 98%   ###
exact_timeMeans_c098 =   [E1215_i3c098[0],              E2025_i3c098[0],            E2430_i3c098[0]*4,            E3645_i3c098[0]]
ga_timeMeans_c098 =      [np.mean(M1215_i3c098[:,0]),   np.mean(M2025_i3c098[:,0]), np.mean(M2430_i3c098[:,0]), np.mean(M3645_i3c098[:,0])]
heur_timeMeans_c098 =    [np.mean(H1215_k5[:,0]),       np.mean(H2025_k7[:,0]),     np.mean(H2430_k6[:,0]),     np.mean(H3645_k6[:,0])]

exact_timeStd_c098 =     [6,                            580,                        800,                         21000]
ga_timeStd_c098 =        [np.std(M1215_i3c098[:,0]),    np.std(M2025_i3c098[:,0]),  np.std(M2430_i3c098[:,0]),  np.std(M3645_i3c098[:,0])]
heur_timeStd_c098 =      [np.std(H1215_k5[:,0]),        np.std(H2025_k7[:,0]),      np.std(H2430_k6[:,0]),      np.std(H3645_k6[:,0])]

###   Consenso de cobertura de 100%   ###
exact_timeMeans_c1 =   [E1215_i3c1[0],              E2025_i3c1[0],              E2430_i3c1[0]*5,              E3645_i3c1[0]]
ga_timeMeans_c1 =      [np.mean(M1215_i3c1[:,0]),   np.mean(M2025_i3c1[:,0]),   np.mean(M2430_i3c1[:,0]),   np.mean(M3645_i3c1[:,0])]
heur_timeMeans_c1 =    [np.mean(H1215_k4[:,0]),     np.mean(H2025_k6[:,0]),     np.mean(H2430_k6[:,0]),     np.mean(H3645_k7[:,0])]

exact_timeStd_c1 =     [5,                          50,                         700,                        18000]
ga_timeStd_c1 =        [np.std(M1215_i3c1[:,0]),    np.std(M2025_i3c1[:,0]),    np.std(M2430_i3c1[:,0]),    np.std(M3645_i3c1[:,0])]
heur_timeStd_c1 =      [np.std(H1215_k4[:,0]),      np.std(H2025_k6[:,0]),      np.std(H2430_k6[:,0]),      np.std(H3645_k7[:,0])]

labels = ['Brum\n1215', 'Brum\n2025', 'Brum\n2430', 'Brum\n3645']
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
axs[0, 0].bar(x - width,  exact_timeMeans_c08,    width, yerr=exact_timeStd_c08,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0, 0].bar(x ,         ga_timeMeans_c08,       width, yerr=ga_timeStd_c08 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0, 0].bar(x + width,  heur_timeMeans_c08,     width, yerr=heur_timeStd_c08 ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[0, 1].bar(x - width,  exact_timeMeans_c09,    width, yerr=exact_timeStd_c09,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0, 1].bar(x ,         ga_timeMeans_c09,       width, yerr=ga_timeStd_c09 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0, 1].bar(x + width,  heur_timeMeans_c09,     width, yerr=heur_timeStd_c09 ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 0].bar(x - width,  exact_timeMeans_c095,    width, yerr=exact_timeStd_c095,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 0].bar(x ,         ga_timeMeans_c095,       width, yerr=ga_timeStd_c095 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 0].bar(x + width,  heur_timeMeans_c095,     width, yerr=heur_timeStd_c095 ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 1].bar(x - width,  exact_timeMeans_c098,    width, yerr=exact_timeStd_c098,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 1].bar(x ,         ga_timeMeans_c098,       width, yerr=ga_timeStd_c098 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 1].bar(x + width,  heur_timeMeans_c098,     width, yerr=heur_timeStd_c098 ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 2].bar(x - width,  exact_timeMeans_c1,    width, yerr=exact_timeStd_c1,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 2].bar(x ,         ga_timeMeans_c1,       width, yerr=ga_timeStd_c1 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 2].bar(x + width,  heur_timeMeans_c1,     width, yerr=heur_timeStd_c1 ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[0, 0].set_ylabel('Time (s)')
axs[1, 0].set_ylabel('Time (s)')
axs[1, 0].set_xticks(x)
axs[1, 1].set_xticks(x)
axs[1, 2].set_xticks(x)
axs[1, 0].set_xticklabels(labels) # rotation=45, fontsize=8
axs[1, 1].set_xticklabels(labels)
axs[1, 2].set_xticklabels(labels)
axs[0, 0].grid(axis='y', alpha=0.4)
axs[0, 1].grid(axis='y', alpha=0.4)
#axs[0, 2].grid(axis='y', alpha=0.4)
axs[1, 0].grid(axis='y', alpha=0.4)
axs[1, 1].grid(axis='y', alpha=0.4)
axs[1, 2].grid(axis='y', alpha=0.4)
axs[0, 0].set_title('Coverage 80%')
axs[0, 1].set_title('Coverage 90%')
axs[1, 0].set_title('Coverage 95%')
axs[1, 1].set_title('Coverage 98%')
axs[1, 2].set_title('Coverage 100%')
axs[0, 2].remove()
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize='medium') #fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
fig.set_size_inches(9.2, 4.8)
fig.tight_layout(pad=0.1)
plt.yscale('log')
plt.show()

####################################################################################################
###   Gráficos de barras médias/desvio padrão dos tempos de processamento para cada Coverage %   ###
####################################################################################################
x = [80, 90, 95, 98, 100]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']
exact_timeMeans = []
ga_timeMeans = []
heur_timeMeans = []
exact_timeStd = []
ga_timeStd = []
heur_timeStd =[]

###   Cenário Brum1215   ###
exact_timeMeans.append(  [E1215_i3c08[0],            E1215_i3c09[0],             E1215_i3c095[0],            E1215_i3c098[0],               E1215_i3c1[0]               ])
ga_timeMeans.append(     [np.mean(M1215_i3c08[:,0]), np.mean(M1215_i3c09[:,0]),  np.mean(M1215_i3c095[:,0]), np.mean(M1215_i3c098[:,0]),    np.mean(M1215_i3c1[:,0])    ])
heur_timeMeans.append(   [np.mean(H1215_k5[:,0]),    np.mean(H1215_k7[:,0]),     np.mean(H1215_k6[:,0]),     np.mean(H1215_k5[:,0]),        np.mean(H1215_k4[:,0])      ])

exact_timeStd.append(    [4,                         5,                          6,                          6,                             5                           ])
ga_timeStd.append(       [np.std(M1215_i3c08[:,0]),  np.std(M1215_i3c09[:,0]),   np.std(M1215_i3c095[:,0]),  np.std(M1215_i3c098[:,0]),     np.std(M1215_i3c1[:,0])     ])
heur_timeStd.append(     [np.std(H1215_k5[:,0]),     np.std(H1215_k7[:,0]),      np.std(H1215_k6[:,0]),      np.std(H1215_k5[:,0]),         np.std(H1215_k4[:,0])       ])

###   Cenário Brum2025   ###
exact_timeMeans.append(  [E2025_i3c08[0],            E2025_i3c09[0],             E2025_i3c095[0],            E2025_i3c098[0],               E2025_i3c1[0]               ])
ga_timeMeans.append(     [np.mean(M2025_i3c08[:,0]), np.mean(M2025_i3c09[:,0]),  np.mean(M2025_i3c095[:,0]), np.mean(M2025_i3c098[:,0]),    np.mean(M2025_i3c1[:,0])    ])
heur_timeMeans.append(   [np.mean(H2025_k6[:,0]),    np.mean(H2025_k6[:,0]),     np.mean(H2025_k7[:,0]),     np.mean(H2025_k7[:,0]),        np.mean(H2025_k6[:,0])      ])

exact_timeStd.append(    [25,                        50,                         32,                         380,                            56                          ])
ga_timeStd.append(       [np.std(M2025_i3c08[:,0]),  np.std(M2025_i3c09[:,0]),   np.std(M2025_i3c095[:,0]),  np.std(M2025_i3c098[:,0]),     np.std(M2025_i3c1[:,0])     ])
heur_timeStd.append(     [np.std(H2025_k6[:,0]),     np.std(H2025_k6[:,0]),      np.std(H2025_k7[:,0]),      np.std(H2025_k7[:,0]),         np.std(H2025_k6[:,0])       ])

###   Cenário Brum2430   ###
exact_timeMeans.append(  [E2430_i3c08[0]*10,            E2430_i3c09[0]*4,           E2430_i3c095[0]*4,           E2430_i3c098[0]*4,         E2430_i3c1[0]*5             ])
ga_timeMeans.append(     [np.mean(M2430_i3c08[:,0]),    np.mean(M2430_i3c09[:,0]),  np.mean(M2430_i3c095[:,0]),  np.mean(M2430_i3c098[:,0]),np.mean(M2430_i3c1[:,0])    ])
heur_timeMeans.append(   [np.mean(H2430_k8[:,0]),       np.mean(H2430_k5[:,0]),     np.mean(H2430_k6[:,0]),      np.mean(H2430_k6[:,0]),    np.mean(H2430_k6[:,0])      ])

exact_timeStd.append(    [120,                          700,                        720,                        820,                        760                         ])
ga_timeStd.append(       [np.std(M2430_i3c08[:,0]),     np.std(M2430_i3c09[:,0]),   np.std(M2430_i3c095[:,0]),  np.std(M2430_i3c098[:,0]),  np.std(M2430_i3c1[:,0])     ])
heur_timeStd.append(     [np.std(H2430_k8[:,0]),        np.std(H2430_k5[:,0]),      np.std(H2430_k6[:,0]),      np.std(H2430_k6[:,0]),      np.std(H2430_k6[:,0])       ])

###   Cenário Brum3645   ###
exact_timeMeans.append(  [E3645_i3c08[0],               E3645_i3c09[0],             E3645_i3c095[0],            E3645_i3c098[0],            E3645_i3c1[0]               ])
ga_timeMeans.append(     [np.mean(M3645_i3c08[:,0]),    np.mean(M3645_i3c09[:,0]),  np.mean(M3645_i3c095[:,0]), np.mean(M3645_i3c098[:,0]), np.mean(M3645_i3c1[:,0])    ])
heur_timeMeans.append(   [np.mean(H3645_k5[:,0]),       np.mean(H3645_k7[:,0]),     np.mean(H3645_k6[:,0]),     np.mean(H3645_k6[:,0]),     np.mean(H3645_k7[:,0])      ])

exact_timeStd.append(    [3000,                         30000,                      82000,                      21000,                      18000                       ])
ga_timeStd.append(       [np.std(M3645_i3c08[:,0]),     np.std(M3645_i3c09[:,0]),   np.std(M3645_i3c095[:,0]),  np.std(M3645_i3c098[:,0]),  np.std(M3645_i3c1[:,0])     ])
heur_timeStd.append(     [np.std(H3645_k5[:,0]),        np.std(H3645_k7[:,0]),      np.std(H3645_k6[:,0]),      np.std(H3645_k6[:,0]),      np.std(H3645_k7[:,0])       ])


fig, axs = plt.subplots(1, 4, sharey=True)
labels = [80, 90, 95, 98, 100]
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
axs[0].bar(x - width,  exact_timeMeans[0],    width, yerr=exact_timeStd[0],  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0].bar(x ,         ga_timeMeans[0],       width, yerr=ga_timeStd[0] ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0].bar(x + width,  heur_timeMeans[0],     width, yerr=heur_timeStd[0] ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1].bar(x - width,  exact_timeMeans[1],    width, yerr=exact_timeStd[1],  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1].bar(x ,         ga_timeMeans[1],       width, yerr=ga_timeStd[1] ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1].bar(x + width,  heur_timeMeans[1],     width, yerr=heur_timeStd[1] ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[2].bar(x - width,  exact_timeMeans[2],    width, yerr=exact_timeStd[2],  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[2].bar(x ,         ga_timeMeans[2],       width, yerr=ga_timeStd[2] ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[2].bar(x + width,  heur_timeMeans[2],     width, yerr=heur_timeStd[2] ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[3].bar(x - width,  exact_timeMeans[3],    width, yerr=exact_timeStd[3],  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[3].bar(x ,         ga_timeMeans[3],       width, yerr=ga_timeStd[3] ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[3].bar(x + width,  heur_timeMeans[3],     width, yerr=heur_timeStd[3] ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[0].grid(axis='y', alpha=0.4)
axs[1].grid(axis='y', alpha=0.4)
axs[2].grid(axis='y', alpha=0.4)
axs[3].grid(axis='y', alpha=0.4)

axs[0].set_title(cenarios[0])
axs[1].set_title(cenarios[1])
axs[2].set_title(cenarios[2])
axs[3].set_title(cenarios[3])

axs[0].set_ylabel('Processing Time (s)')

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Coverage Consensus (%)")

axs[0].set_xticks(x)
axs[1].set_xticks(x)
axs[2].set_xticks(x)
axs[3].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[1].set_xticklabels(labels)
axs[2].set_xticklabels(labels)
axs[3].set_xticklabels(labels)

axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
axs[0].set_yscale('log')
plt.subplots_adjust(left=0.05, bottom=0.15)
plt.show()

################################################################################
###   Gráfico [Grau de Sobreposição x Consenso de Cobertura] (por cenário)   ###
################################################################################
x = [80, 90, 95, 98, 100]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']
exact_fitValues = []
ga_fitMeans = []
heur_fitMeans = []
ga_fitStd = []
heur_fitStd =[]

###   Cenário Brum1215   ###
exact_fitValues.append( [E1215_i3c08[1]*(-100),            E1215_i3c09[1]*(-100),             E1215_i3c095[1]*(-100),            E1215_i3c098[1]*(-100),            E1215_i3c1[1]*(-100)                ])
ga_fitMeans.append(     [np.mean(M1215_i3c08[:,1])*(-100), np.mean(M1215_i3c09[:,1])*(-100),  np.mean(M1215_i3c095[:,1])*(-100), np.mean(M1215_i3c098[:,1])*(-100), np.mean(M1215_i3c1[:,1])*(-100)     ])
heur_fitMeans.append(   [np.mean(H1215_k5[:,1])*(-100),    np.mean(H1215_k7[:,1])*(-100),     np.mean(H1215_k6[:,1])*(-100),     np.mean(H1215_k5[:,1])*(-100),     np.mean(H1215_k4[:,1])*(-100)       ])

ga_fitStd.append(       [np.std(M1215_i3c08[:,1])*(-100),  np.std(M1215_i3c09[:,1])*(-100),   np.std(M1215_i3c095[:,1])*(-100),  np.std(M1215_i3c098[:,1])*(-100),  np.std(M1215_i3c1[:,1])*(-100) ])
heur_fitStd.append(     [np.std(H1215_k5[:,1])*(-100),     np.std(H1215_k7[:,1])*(-100),      np.std(H1215_k6[:,1])*(-100),      np.std(H1215_k5[:,1])*(-100),      np.std(H1215_k4[:,1])*(-100)   ])

###   Cenário Brum2025   ###
exact_fitValues.append( [E2025_i3c08[1]*(-100),            E2025_i3c09[1]*(-100),             E2025_i3c095[1]*(-100),            E2025_i3c098[1]*(-100),            E2025_i3c1[1]*(-100)           ])
ga_fitMeans.append(     [np.mean(M2025_i3c08[:,1])*(-100), np.mean(M2025_i3c09[:,1])*(-100),  np.mean(M2025_i3c095[:,1])*(-100), np.mean(M2025_i3c098[:,1])*(-100), np.mean(M2025_i3c1[:,1])*(-100)])
heur_fitMeans.append(   [np.mean(H2025_k6[:,1])*(-100),    np.mean(H2025_k6[:,1])*(-100),     np.mean(H2025_k7[:,1])*(-100),     np.mean(H2025_k7[:,1])*(-100),     np.mean(H2025_k6[:,1])*(-100)  ])

ga_fitStd.append(       [np.std(M2025_i3c08[:,1])*(-100),  np.std(M2025_i3c09[:,1])*(-100),   np.std(M2025_i3c095[:,1])*(-100),  np.std(M2025_i3c098[:,1])*(-100),  np.std(M2025_i3c1[:,1])*(-100) ])
heur_fitStd.append(     [np.std(H2025_k6[:,1])*(-100),     np.std(H2025_k6[:,1])*(-100),      np.std(H2025_k7[:,1])*(-100),      np.std(H2025_k7[:,1])*(-100),      np.std(H2025_k6[:,1])*(-100)   ])

###   Cenário Brum2430   ###
exact_fitValues.append( [E2430_i3c08[1]*(-100),            E2430_i3c09[1]*(-100),             E2430_i3c095[1]*(-100),            E2430_i3c098[1]*(-100),            E2430_i3c1[1]*(-100)           ])
ga_fitMeans.append(     [np.mean(M2430_i3c08[:,1])*(-100), np.mean(M2430_i3c09[:,1])*(-100),  np.mean(M2430_i3c095[:,1])*(-100), np.mean(M2430_i3c098[:,1])*(-100), np.mean(M2430_i3c1[:,1])*(-100)])
heur_fitMeans.append(   [np.mean(H2430_k8[:,1])*(-100),    np.mean(H2430_k5[:,1])*(-100),     np.mean(H2430_k6[:,1])*(-100),     np.mean(H2430_k6[:,1])*(-100),     np.mean(H2430_k6[:,1])*(-100)  ])

ga_fitStd.append(       [np.std(M2430_i3c08[:,1])*(-100),  np.std(M2430_i3c09[:,1])*(-100),   np.std(M2430_i3c095[:,1])*(-100),  np.std(M2430_i3c098[:,1])*(-100),  np.std(M2430_i3c1[:,1])*(-100) ])
heur_fitStd.append(     [np.std(H2430_k8[:,1])*(-100),     np.std(H2430_k5[:,1])*(-100),      np.std(H2430_k6[:,1])*(-100),      np.std(H2430_k6[:,1])*(-100),      np.std(H2430_k6[:,1])*(-100)   ])

###   Cenário Brum3645   ###
exact_fitValues.append( [E3645_i3c08[1]*(-100),            E3645_i3c09[1]*(-100),             E3645_i3c095[1]*(-100),            E3645_i3c098[1]*(-100),            E3645_i3c1[1]*(-100)           ])
ga_fitMeans.append(     [np.mean(M3645_i3c08[:,1])*(-100), np.mean(M3645_i3c09[:,1])*(-100),  np.mean(M3645_i3c095[:,1])*(-100), np.mean(M3645_i3c098[:,1])*(-100), np.mean(M3645_i3c1[:,1])*(-100)])
heur_fitMeans.append(   [np.mean(H3645_k5[:,1])*(-100),    np.mean(H3645_k7[:,1])*(-100),     np.mean(H3645_k6[:,1])*(-100),     np.mean(H3645_k6[:,1])*(-100),     np.mean(H3645_k7[:,1])*(-100)  ])

ga_fitStd.append(       [np.std(M3645_i3c08[:,1])*(-100),  np.std(M3645_i3c09[:,1])*(-100),   np.std(M3645_i3c095[:,1])*(-100),  np.std(M3645_i3c098[:,1])*(-100),  np.std(M3645_i3c1[:,1])*(-100) ])
heur_fitStd.append(     [np.std(H3645_k5[:,1])*(-100),     np.std(H3645_k7[:,1])*(-100),      np.std(H3645_k6[:,1])*(-100),      np.std(H3645_k6[:,1])*(-100),      np.std(H3645_k7[:,1])*(-100)   ])


fig, axs = plt.subplots(1, 4, sharey=True)
axs[0].plot(x, exact_fitValues[0], marker='s', color='r', label="E-ALLOCATOR")
axs[0].errorbar(x, ga_fitMeans[0],     ga_fitStd[0],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[0].errorbar(x, heur_fitMeans[0],   heur_fitStd[0], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[1].plot(x, exact_fitValues[1], marker='s', color='r', label="E-ALLOCATOR")
axs[1].errorbar(x, ga_fitMeans[1],     ga_fitStd[1],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[1].errorbar(x, heur_fitMeans[1],   heur_fitStd[1], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[2].plot(x, exact_fitValues[2], marker='s', color='r', label="E-ALLOCATOR")
axs[2].errorbar(x, ga_fitMeans[2],     ga_fitStd[2],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[2].errorbar(x, heur_fitMeans[2],   heur_fitStd[2], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[3].plot(x, exact_fitValues[3], marker='s', color='r', label="E-ALLOCATOR")
axs[3].errorbar(x, ga_fitMeans[3],     ga_fitStd[3],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[3].errorbar(x, heur_fitMeans[3],   heur_fitStd[3], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[0].grid(alpha=0.4)
axs[1].grid(alpha=0.4)
axs[2].grid(alpha=0.4)
axs[3].grid(alpha=0.4)

axs[0].set_title(cenarios[0])
axs[1].set_title(cenarios[1])
axs[2].set_title(cenarios[2])
axs[3].set_title(cenarios[3])

axs[0].set_ylim([0, 60])
axs[1].set_ylim([0, 60])
axs[2].set_ylim([0, 60])
axs[3].set_ylim([0, 60])

axs[0].set_ylabel('Degree of Overlap (%)')

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Coverage Consensus (%)")

axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
plt.subplots_adjust(left=0.04, bottom=0.15)
plt.show()

#############################################################################
###   Gráfico [Número de eNodeBs x Consenso de Cobertura] (por cenário)   ###
#############################################################################
x = [80, 90, 95, 98, 100]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']
exact_NreNBs = []
ga_NreNBsMeans = []
heur_NreNBsMeans = []
ga_NreNBsStd = []
heur_NreNBsStd =[]

###   Cenário Brum1215   ###
exact_NreNBs.append(       [E1215_i3c08[2],            E1215_i3c09[2],             E1215_i3c095[2],            E1215_i3c098[2],            E1215_i3c1[2]])
ga_NreNBsMeans.append(     [np.mean(M1215_i3c08[:,2]), np.mean(M1215_i3c09[:,2]),  np.mean(M1215_i3c095[:,2]), np.mean(M1215_i3c098[:,2]), np.mean(M1215_i3c1[:,2])])
#heur_NreNBsMeans.append(   [np.mean(H1215_k5[:,2]),    np.mean(H1215_k7[:,2]),     np.mean(H1215_k6[:,2]),     np.mean(H1215_k5[:,2]),     np.mean(H1215_k4[:,2])])

ga_NreNBsStd.append(       [np.std(M1215_i3c08[:,2]),  np.std(M1215_i3c09[:,2]),   np.std(M1215_i3c095[:,2]),  np.std(M1215_i3c098[:,2]),  np.std(M1215_i3c1[:,2])])
#heur_NreNBsStd.append(     [np.std(H1215_k5[:,2]),     np.std(H1215_k7[:,2]),      np.std(H1215_k6[:,2]),      np.std(H1215_k5[:,2]),      np.std(H1215_k4[:,2])])

###   Cenário Brum2025   ###
exact_NreNBs.append(       [E2025_i3c08[2],            E2025_i3c09[2],             E2025_i3c095[2],            E2025_i3c098[2],            E2025_i3c1[2]])
ga_NreNBsMeans.append(     [np.mean(M2025_i3c08[:,2]), np.mean(M2025_i3c09[:,2]),  np.mean(M2025_i3c095[:,2]), np.mean(M2025_i3c098[:,2]), np.mean(M2025_i3c1[:,2])])
#heur_NreNBsMeans.append(   [np.mean(H2025_k6[:,2]),    np.mean(H2025_k6[:,2]),     np.mean(H2025_k7[:,2]),     np.mean(H2025_k7[:,2]),     np.mean(H2025_k6[:,2])])

ga_NreNBsStd.append(       [np.std(M2025_i3c08[:,2]),  np.std(M2025_i3c09[:,2]),   np.std(M2025_i3c095[:,2]),  np.std(M2025_i3c098[:,2]),  np.std(M2025_i3c1[:,2])])
#heur_NreNBsStd.append(     [np.std(H2025_k6[:,2]),     np.std(H2025_k6[:,2]),      np.std(H2025_k7[:,2]),      np.std(H2025_k7[:,2]),      np.std(H2025_k6[:,2])])

###   Cenário Brum2430   ###
exact_NreNBs.append(       [E2430_i3c08[2],            E2430_i3c09[2],             E2430_i3c095[2],            E2430_i3c098[2],            E2430_i3c1[2]])
ga_NreNBsMeans.append(     [np.mean(M2430_i3c08[:,2]), np.mean(M2430_i3c09[:,2]),  np.mean(M2430_i3c095[:,2]), np.mean(M2430_i3c098[:,2]), np.mean(M2430_i3c1[:,2])])
#heur_NreNBsMeans.append(   [np.mean(H2430_k8[:,2]),    np.mean(H2430_k5[:,2]),     np.mean(H2430_k6[:,2]),     np.mean(H2430_k6[:,2]),     np.mean(H2430_k6[:,2])])

ga_NreNBsStd.append(       [np.std(M2430_i3c08[:,2]),  np.std(M2430_i3c09[:,2]),   np.std(M2430_i3c095[:,2]),  np.std(M2430_i3c098[:,2]),  np.std(M2430_i3c1[:,2])])
#heur_NreNBsStd.append(     [np.std(H2430_k8[:,2]),     np.std(H2430_k5[:,2]),      np.std(H2430_k6[:,2]),      np.std(H2430_k6[:,2]),      np.std(H2430_k6[:,2])])

###   Cenário Brum3645   ###
exact_NreNBs.append(       [E3645_i3c08[2],            E3645_i3c09[2],             E3645_i3c095[2],            E3645_i3c098[2],            E3645_i3c1[2]])
ga_NreNBsMeans.append(     [np.mean(M3645_i3c08[:,2]), np.mean(M3645_i3c09[:,2]),  np.mean(M3645_i3c095[:,2]), np.mean(M3645_i3c098[:,2]), np.mean(M3645_i3c1[:,2])])
#heur_NreNBsMeans.append(   [np.mean(H3645_k5[:,2]),    np.mean(H3645_k7[:,2]),     np.mean(H3645_k6[:,2]),     np.mean(H3645_k6[:,2]),     np.mean(H3645_k7[:,2])])

ga_NreNBsStd.append(       [np.std(M3645_i3c08[:,2]),  np.std(M3645_i3c09[:,2]),   np.std(M3645_i3c095[:,2]),  np.std(M3645_i3c098[:,2]),  np.std(M3645_i3c1[:,2])])
#heur_NreNBsStd.append(     [np.std(H3645_k5[:,2]),     np.std(H3645_k7[:,2]),      np.std(H3645_k6[:,2]),      np.std(H3645_k6[:,2]),      np.std(H3645_k7[:,2])])

fig, axs = plt.subplots(1, 4, sharey=True)
axs[0].plot(x, exact_NreNBs[0], marker='s', color='r', label="E-ALLOCATOR")
axs[0].errorbar(x, ga_NreNBsMeans[0],     ga_NreNBsStd[0],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
#axs[0].errorbar(x, heur_NreNBsMeans[0],   heur_NreNBsStd[0], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[1].plot(x, exact_NreNBs[1], marker='s', color='r', label="E-ALLOCATOR")
axs[1].errorbar(x, ga_NreNBsMeans[1],     ga_NreNBsStd[1],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
#axs[1].errorbar(x, heur_NreNBsMeans[1],   heur_NreNBsStd[1], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[2].plot(x, exact_NreNBs[2], marker='s', color='r', label="E-ALLOCATOR")
axs[2].errorbar(x, ga_NreNBsMeans[2],     ga_NreNBsStd[2],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
#axs[2].errorbar(x, heur_NreNBsMeans[2],   heur_NreNBsStd[2], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[3].plot(x, exact_NreNBs[3], marker='s', color='r', label="E-ALLOCATOR")
axs[3].errorbar(x, ga_NreNBsMeans[3],     ga_NreNBsStd[3],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
#axs[3].errorbar(x, heur_NreNBsMeans[3],   heur_NreNBsStd[3], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[0].grid(alpha=0.4)
axs[1].grid(alpha=0.4)
axs[2].grid(alpha=0.4)
axs[3].grid(alpha=0.4)
#axs[4].remove()

axs[0].set_title(cenarios[0])
axs[1].set_title(cenarios[1])
axs[2].set_title(cenarios[2])
axs[3].set_title(cenarios[3])

axs[0].set_ylim([0, 10])
axs[1].set_ylim([0, 10])
axs[2].set_ylim([0, 10])
axs[3].set_ylim([0, 10])

axs[0].set_ylabel('Number of eNodeBs deployed')

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Coverage Consensus (%)")

axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
plt.subplots_adjust(left=0.04, bottom=0.15)
plt.show()

#############################################################################
###   Gráfico [Área Coberta x Consenso de Cobertura] (por cenário)   ###
#############################################################################
x = [80, 90, 95, 98, 100]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']
exact_Area = []
ga_AreaMeans = []
heur_AreaMeans = []
ga_AreaStd = []
heur_AreaStd =[]

###   Cenário Brum1215   ###
exact_Area.append(       [E1215_i3c08[4],            E1215_i3c09[4],             E1215_i3c095[4],            E1215_i3c098[4],            E1215_i3c1[4]])
ga_AreaMeans.append(     [np.mean(M1215_i3c08[:,4]), np.mean(M1215_i3c09[:,4]),  np.mean(M1215_i3c095[:,4]), np.mean(M1215_i3c098[:,4]), np.mean(M1215_i3c1[:,4])])
heur_AreaMeans.append(   [np.mean(H1215_k5[:,4]),    np.mean(H1215_k7[:,4]),     np.mean(H1215_k6[:,4]),     np.mean(H1215_k5[:,4]),     np.mean(H1215_k4[:,4])])

ga_AreaStd.append(       [np.std(M1215_i3c08[:,4]),  np.std(M1215_i3c09[:,4]),   np.std(M1215_i3c095[:,4]),  np.std(M1215_i3c098[:,4]),  np.std(M1215_i3c1[:,4])])
heur_AreaStd.append(     [np.std(H1215_k5[:,4]),     np.std(H1215_k7[:,4]),      np.std(H1215_k6[:,4]),      np.std(H1215_k5[:,4]),      np.std(H1215_k4[:,4])])

###   Cenário Brum2025   ###
exact_Area.append(       [E2025_i3c08[4],            E2025_i3c09[4],             E2025_i3c095[4],            E2025_i3c098[4],            E2025_i3c1[4]])
ga_AreaMeans.append(     [np.mean(M2025_i3c08[:,4]), np.mean(M2025_i3c09[:,4]),  np.mean(M2025_i3c095[:,4]), np.mean(M2025_i3c098[:,4]), np.mean(M2025_i3c1[:,4])])
heur_AreaMeans.append(   [np.mean(H2025_k6[:,4]),    np.mean(H2025_k6[:,4]),     np.mean(H2025_k7[:,4]),     np.mean(H2025_k7[:,4]),     np.mean(H2025_k6[:,4])])

ga_AreaStd.append(       [np.std(M2025_i3c08[:,4]),  np.std(M2025_i3c09[:,4]),   np.std(M2025_i3c095[:,4]),  np.std(M2025_i3c098[:,4]),  np.std(M2025_i3c1[:,4])])
heur_AreaStd.append(     [np.std(H2025_k6[:,4]),     np.std(H2025_k6[:,4]),      np.std(H2025_k7[:,4]),      np.std(H2025_k7[:,4]),      np.std(H2025_k6[:,4])])

###   Cenário Brum2430   ###
exact_Area.append(       [E2430_i3c08[4],            E2430_i3c09[4],             E2430_i3c095[4],            E2430_i3c098[4],            E2430_i3c1[4]])
ga_AreaMeans.append(     [np.mean(M2430_i3c08[:,4]), np.mean(M2430_i3c09[:,4]),  np.mean(M2430_i3c095[:,4]), np.mean(M2430_i3c098[:,4]), np.mean(M2430_i3c1[:,4])])
heur_AreaMeans.append(   [np.mean(H2430_k8[:,4]),    np.mean(H2430_k5[:,4]),     np.mean(H2430_k6[:,4]),     np.mean(H2430_k6[:,4]),     np.mean(H2430_k6[:,4])])

ga_AreaStd.append(       [np.std(M2430_i3c08[:,4]),  np.std(M2430_i3c09[:,4]),   np.std(M2430_i3c095[:,4]),  np.std(M2430_i3c098[:,4]),  np.std(M2430_i3c1[:,4])])
heur_AreaStd.append(     [np.std(H2430_k8[:,4]),     np.std(H2430_k5[:,4]),      np.std(H2430_k6[:,4]),      np.std(H2430_k6[:,4]),      np.std(H2430_k6[:,4])])

###   Cenário Brum3645   ###
exact_Area.append(       [E3645_i3c08[4],            E3645_i3c09[4],             E3645_i3c095[4],            E3645_i3c098[4],            E3645_i3c1[4]])
ga_AreaMeans.append(     [np.mean(M3645_i3c08[:,4]), np.mean(M3645_i3c09[:,4]),  np.mean(M3645_i3c095[:,4]), np.mean(M3645_i3c098[:,4]), np.mean(M3645_i3c1[:,4])])
heur_AreaMeans.append(   [np.mean(H3645_k5[:,4]),    np.mean(H3645_k7[:,4]),     np.mean(H3645_k6[:,4]),     np.mean(H3645_k6[:,4]),     np.mean(H3645_k7[:,4])])

ga_AreaStd.append(       [np.std(M3645_i3c08[:,4]),  np.std(M3645_i3c09[:,4]),   np.std(M3645_i3c095[:,4]),  np.std(M3645_i3c098[:,4]),  np.std(M3645_i3c1[:,4])])
heur_AreaStd.append(     [np.std(H3645_k5[:,4]),     np.std(H3645_k7[:,4]),      np.std(H3645_k6[:,4]),      np.std(H3645_k6[:,4]),      np.std(H3645_k7[:,4])])

fig, axs = plt.subplots(1, 4, sharey=True)
axs[0].plot(x, exact_Area[0], marker='s', color='r', label="E-ALLOCATOR")
axs[0].errorbar(x, ga_AreaMeans[0],     ga_AreaStd[0],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[0].errorbar(x, heur_AreaMeans[0],   heur_AreaStd[0], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[1].plot(x, exact_Area[1], marker='s', color='r', label="E-ALLOCATOR")
axs[1].errorbar(x, ga_AreaMeans[1],     ga_AreaStd[1],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[1].errorbar(x, heur_AreaMeans[1],   heur_AreaStd[1], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[2].plot(x, exact_Area[2], marker='s', color='r', label="E-ALLOCATOR")
axs[2].errorbar(x, ga_AreaMeans[2],     ga_AreaStd[2],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[2].errorbar(x, heur_AreaMeans[2],   heur_AreaStd[2], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[3].plot(x, exact_Area[3], marker='s', color='r', label="E-ALLOCATOR")
axs[3].errorbar(x, ga_AreaMeans[3],     ga_AreaStd[3],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[3].errorbar(x, heur_AreaMeans[3],   heur_AreaStd[3], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="Heuristic")

axs[0].grid(alpha=0.4)
axs[1].grid(alpha=0.4)
axs[2].grid(alpha=0.4)
axs[3].grid(alpha=0.4)

axs[0].set_title(cenarios[0])
axs[1].set_title(cenarios[1])
axs[2].set_title(cenarios[2])
axs[3].set_title(cenarios[3])

axs[0].set_ylim([0, 30])
axs[1].set_ylim([0, 30])
axs[2].set_ylim([0, 30])
axs[3].set_ylim([0, 30])

axs[0].set_ylabel('Coverage Area (km\u00b2)')

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Coverage Consensus (%)")

axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
plt.subplots_adjust(left=0.04, bottom=0.15)
plt.show()

###########################################################################
###   Gráfico [Área Sobreposta x Consenso de Cobertura] (por cenário)   ###
###########################################################################
x = [80, 90, 95, 98, 100]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645']

E_semOverlapMeans  = []
E_overlap2Means    = []
E_overlap3Means    = []
E_overlap4Means    = []
E_overlap5Means    = []
E_overlap6Means    = []
E_overlap7Means    = []
E_overlap8Means    = []
E_overlap9Means    = []
E_semOverlapStd    = []
E_overlap2Std      = []
E_overlap3Std      = []
E_overlap4Std      = []
E_overlap5Std      = []
E_overlap6Std      = []
E_overlap7Std      = []
E_overlap8Std      = []
E_overlap9Std      = []

M_semOverlapMeans  = []
M_overlap2Means    = []
M_overlap3Means    = []
M_overlap4Means    = []
M_overlap5Means    = []
M_overlap6Means    = []
M_overlap7Means    = []
M_overlap8Means    = []
M_overlap9Means    = []
M_semOverlapStd    = []
M_overlap2Std      = []
M_overlap3Std      = []
M_overlap4Std      = []
M_overlap5Std      = []
M_overlap6Std      = []
M_overlap7Std      = []
M_overlap8Std      = []
M_overlap9Std      = []

H_semOverlapMeans  = []
H_overlap2Means    = []
H_overlap3Means    = []
H_overlap4Means    = []
H_overlap5Means    = []
H_overlap6Means    = []
H_overlap7Means    = []
H_overlap8Means    = []
H_overlap9Means    = []
H_semOverlapStd    = []
H_overlap2Std      = []
H_overlap3Std      = []
H_overlap4Std      = []
H_overlap5Std      = []
H_overlap6Std      = []
H_overlap7Std      = []
H_overlap8Std      = []
H_overlap9Std      = []

###   Cenário Brum1215   ###
E_semOverlapMeans.append([ E1215_i3c08[4]-  E1215_i3c08[5]-     E1215_i3c08[6]-     E1215_i3c08[7]-     E1215_i3c08[8]-     E1215_i3c08[9]-     E1215_i3c08[10]-    E1215_i3c08[11]-    E1215_i3c08[12],
                           E1215_i3c09[4]-  E1215_i3c09[5]-     E1215_i3c09[6]-     E1215_i3c09[7]-     E1215_i3c09[8]-     E1215_i3c09[9]-     E1215_i3c09[10]-    E1215_i3c09[11]-    E1215_i3c09[12],
                           E1215_i3c095[4]- E1215_i3c095[5]-    E1215_i3c095[6]-    E1215_i3c095[7]-    E1215_i3c095[8]-    E1215_i3c095[9]-    E1215_i3c095[10]-   E1215_i3c095[11]-   E1215_i3c095[12],
                           E1215_i3c098[4]- E1215_i3c098[5]-    E1215_i3c098[6]-    E1215_i3c098[7]-    E1215_i3c098[8]-    E1215_i3c098[9]-    E1215_i3c098[10]-   E1215_i3c098[11]-   E1215_i3c098[12],
                           E1215_i3c1[4]-   E1215_i3c1[5]-      E1215_i3c1[6]-      E1215_i3c1[7]-      E1215_i3c1[8]-      E1215_i3c1[9]-      E1215_i3c1[10]-     E1215_i3c1[11]-     E1215_i3c1[12]   ])
E_overlap2Means.append([   E1215_i3c08[5],  E1215_i3c09[5],     E1215_i3c095[5],    E1215_i3c098[5],    E1215_i3c1[5]  ])
E_overlap3Means.append([   E1215_i3c08[6],  E1215_i3c09[6],     E1215_i3c095[6],    E1215_i3c098[6],    E1215_i3c1[6]  ])
E_overlap4Means.append([   E1215_i3c08[7],  E1215_i3c09[7],     E1215_i3c095[7],    E1215_i3c098[7],    E1215_i3c1[7]  ])
E_overlap5Means.append([   E1215_i3c08[8],  E1215_i3c09[8],     E1215_i3c095[8],    E1215_i3c098[8],    E1215_i3c1[8]  ])
E_overlap6Means.append([   E1215_i3c08[9],  E1215_i3c09[9],     E1215_i3c095[9],    E1215_i3c098[9],    E1215_i3c1[9]  ])
E_overlap7Means.append([   E1215_i3c08[10], E1215_i3c09[10],    E1215_i3c095[10],   E1215_i3c098[10],   E1215_i3c1[10] ])
E_overlap8Means.append([   E1215_i3c08[11], E1215_i3c09[11],    E1215_i3c095[11],   E1215_i3c098[11],   E1215_i3c1[11] ])
E_overlap9Means.append([   E1215_i3c08[12], E1215_i3c09[12],    E1215_i3c095[12],   E1215_i3c098[12],   E1215_i3c1[12] ])

M_semOverlapMeans.append([ np.mean(M1215_i3c08[:,4])-  np.mean(M1215_i3c08[:,5])-     np.mean(M1215_i3c08[:,6])-     np.mean(M1215_i3c08[:,7])-     np.mean(M1215_i3c08[:,8])-     np.mean(M1215_i3c08[:,9])-     np.mean(M1215_i3c08[:,10])-    np.mean(M1215_i3c08[:,11])-    np.mean(M1215_i3c08[:,12]),
                           np.mean(M1215_i3c09[:,4])-  np.mean(M1215_i3c09[:,5])-     np.mean(M1215_i3c09[:,6])-     np.mean(M1215_i3c09[:,7])-     np.mean(M1215_i3c09[:,8])-     np.mean(M1215_i3c09[:,9])-     np.mean(M1215_i3c09[:,10])-    np.mean(M1215_i3c09[:,11])-    np.mean(M1215_i3c09[:,12]),
                           np.mean(M1215_i3c095[:,4])- np.mean(M1215_i3c095[:,5])-    np.mean(M1215_i3c095[:,6])-    np.mean(M1215_i3c095[:,7])-    np.mean(M1215_i3c095[:,8])-    np.mean(M1215_i3c095[:,9])-    np.mean(M1215_i3c095[:,10])-   np.mean(M1215_i3c095[:,11])-   np.mean(M1215_i3c095[:,12]),
                           np.mean(M1215_i3c098[:,4])- np.mean(M1215_i3c098[:,5])-    np.mean(M1215_i3c098[:,6])-    np.mean(M1215_i3c098[:,7])-    np.mean(M1215_i3c098[:,8])-    np.mean(M1215_i3c098[:,9])-    np.mean(M1215_i3c098[:,10])-   np.mean(M1215_i3c098[:,11])-   np.mean(M1215_i3c098[:,12]),
                           np.mean(M1215_i3c1[:,4])-   np.mean(M1215_i3c1[:,5])-      np.mean(M1215_i3c1[:,6])-      np.mean(M1215_i3c1[:,7])-      np.mean(M1215_i3c1[:,8])-      np.mean(M1215_i3c1[:,9])-      np.mean(M1215_i3c1[:,10])-     np.mean(M1215_i3c1[:,11])-     np.mean(M1215_i3c1[:,12])   ])
M_overlap2Means.append([   np.mean(M1215_i3c08[:,5]),  np.mean(M1215_i3c09[:,5]),     np.mean(M1215_i3c095[:,5]),    np.mean(M1215_i3c098[:,5]),    np.mean(M1215_i3c1[:,5])  ])
M_overlap3Means.append([   np.mean(M1215_i3c08[:,6]),  np.mean(M1215_i3c09[:,6]),     np.mean(M1215_i3c095[:,6]),    np.mean(M1215_i3c098[:,6]),    np.mean(M1215_i3c1[:,6])  ])
M_overlap4Means.append([   np.mean(M1215_i3c08[:,7]),  np.mean(M1215_i3c09[:,7]),     np.mean(M1215_i3c095[:,7]),    np.mean(M1215_i3c098[:,7]),    np.mean(M1215_i3c1[:,7])  ])
M_overlap5Means.append([   np.mean(M1215_i3c08[:,8]),  np.mean(M1215_i3c09[:,8]),     np.mean(M1215_i3c095[:,8]),    np.mean(M1215_i3c098[:,8]),    np.mean(M1215_i3c1[:,8])  ])
M_overlap6Means.append([   np.mean(M1215_i3c08[:,9]),  np.mean(M1215_i3c09[:,9]),     np.mean(M1215_i3c095[:,9]),    np.mean(M1215_i3c098[:,9]),    np.mean(M1215_i3c1[:,9])  ])
M_overlap7Means.append([   np.mean(M1215_i3c08[:,10]), np.mean(M1215_i3c09[:,10]),    np.mean(M1215_i3c095[:,10]),   np.mean(M1215_i3c098[:,10]),   np.mean(M1215_i3c1[:,10]) ])
M_overlap8Means.append([   np.mean(M1215_i3c08[:,11]), np.mean(M1215_i3c09[:,11]),    np.mean(M1215_i3c095[:,11]),   np.mean(M1215_i3c098[:,11]),   np.mean(M1215_i3c1[:,11]) ])
M_overlap9Means.append([   np.mean(M1215_i3c08[:,12]), np.mean(M1215_i3c09[:,12]),    np.mean(M1215_i3c095[:,12]),   np.mean(M1215_i3c098[:,12]),   np.mean(M1215_i3c1[:,12]) ])
M_semOverlapStd.append([   np.std(M1215_i3c08[:,4])-   np.std(M1215_i3c08[:,5])-      np.std(M1215_i3c08[:,6])-      np.std(M1215_i3c08[:,7])-      np.std(M1215_i3c08[:,8])-     np.std(M1215_i3c08[:,9])-     np.std(M1215_i3c08[:,10])-    np.std(M1215_i3c08[:,11])-    np.std(M1215_i3c08[:,12]),
                           np.std(M1215_i3c09[:,4])-   np.std(M1215_i3c09[:,5])-      np.std(M1215_i3c09[:,6])-      np.std(M1215_i3c09[:,7])-      np.std(M1215_i3c09[:,8])-     np.std(M1215_i3c09[:,9])-     np.std(M1215_i3c09[:,10])-    np.std(M1215_i3c09[:,11])-    np.std(M1215_i3c09[:,12]),
                           np.std(M1215_i3c095[:,4])-  np.std(M1215_i3c095[:,5])-     np.std(M1215_i3c095[:,6])-     np.std(M1215_i3c095[:,7])-     np.std(M1215_i3c095[:,8])-    np.std(M1215_i3c095[:,9])-    np.std(M1215_i3c095[:,10])-   np.std(M1215_i3c095[:,11])-   np.std(M1215_i3c095[:,12]),
                           np.std(M1215_i3c098[:,4])-  np.std(M1215_i3c098[:,5])-     np.std(M1215_i3c098[:,6])-     np.std(M1215_i3c098[:,7])-     np.std(M1215_i3c098[:,8])-    np.std(M1215_i3c098[:,9])-    np.std(M1215_i3c098[:,10])-   np.std(M1215_i3c098[:,11])-   np.std(M1215_i3c098[:,12]),
                           np.std(M1215_i3c1[:,4])-    np.std(M1215_i3c1[:,5])-       np.std(M1215_i3c1[:,6])-       np.std(M1215_i3c1[:,7])-       np.std(M1215_i3c1[:,8])-      np.std(M1215_i3c1[:,9])-      np.std(M1215_i3c1[:,10])-     np.std(M1215_i3c1[:,11])-     np.std(M1215_i3c1[:,12])   ])
M_overlap2Std.append([     np.std(M1215_i3c08[:,5]),   np.std(M1215_i3c09[:,5]),      np.std(M1215_i3c095[:,5]),     np.std(M1215_i3c098[:,5]),     np.std(M1215_i3c1[:,5])  ])
M_overlap3Std.append([     np.std(M1215_i3c08[:,6]),   np.std(M1215_i3c09[:,6]),      np.std(M1215_i3c095[:,6]),     np.std(M1215_i3c098[:,6]),     np.std(M1215_i3c1[:,6])  ])
M_overlap4Std.append([     np.std(M1215_i3c08[:,7]),   np.std(M1215_i3c09[:,7]),      np.std(M1215_i3c095[:,7]),     np.std(M1215_i3c098[:,7]),     np.std(M1215_i3c1[:,7])  ])
M_overlap5Std.append([     np.std(M1215_i3c08[:,8]),   np.std(M1215_i3c09[:,8]),      np.std(M1215_i3c095[:,8]),     np.std(M1215_i3c098[:,8]),     np.std(M1215_i3c1[:,8])  ])
M_overlap6Std.append([     np.std(M1215_i3c08[:,9]),   np.std(M1215_i3c09[:,9]),      np.std(M1215_i3c095[:,9]),     np.std(M1215_i3c098[:,9]),     np.std(M1215_i3c1[:,9])  ])
M_overlap7Std.append([     np.std(M1215_i3c08[:,10]),  np.std(M1215_i3c09[:,10]),     np.std(M1215_i3c095[:,10]),    np.std(M1215_i3c098[:,10]),    np.std(M1215_i3c1[:,10]) ])
M_overlap8Std.append([     np.std(M1215_i3c08[:,11]),  np.std(M1215_i3c09[:,11]),     np.std(M1215_i3c095[:,11]),    np.std(M1215_i3c098[:,11]),    np.std(M1215_i3c1[:,11]) ])
M_overlap9Std.append([     np.std(M1215_i3c08[:,12]),  np.std(M1215_i3c09[:,12]),     np.std(M1215_i3c095[:,12]),    np.std(M1215_i3c098[:,12]),    np.std(M1215_i3c1[:,12]) ])

H_semOverlapMeans.append([ np.mean(H1215_k5[:,4])-  np.mean(H1215_k5[:,5])-     np.mean(H1215_k5[:,6])-     np.mean(H1215_k5[:,7])-    np.mean(H1215_k5[:,8])-     np.mean(H1215_k5[:,9])-     np.mean(H1215_k5[:,10])-    np.mean(H1215_k5[:,11])-    np.mean(H1215_k5[:,12]),
                           np.mean(H1215_k7[:,4])-  np.mean(H1215_k7[:,5])-     np.mean(H1215_k7[:,6])-     np.mean(H1215_k7[:,7])-    np.mean(H1215_k7[:,8])-     np.mean(H1215_k7[:,9])-     np.mean(H1215_k7[:,10])-    np.mean(H1215_k7[:,11])-    np.mean(H1215_k7[:,12]),
                           np.mean(H1215_k6[:,4])-  np.mean(H1215_k6[:,5])-     np.mean(H1215_k6[:,6])-     np.mean(H1215_k6[:,7])-    np.mean(H1215_k6[:,8])-     np.mean(H1215_k6[:,9])-     np.mean(H1215_k6[:,10])-    np.mean(H1215_k6[:,11])-    np.mean(H1215_k6[:,12]),
                           np.mean(H1215_k5[:,4])-  np.mean(H1215_k5[:,5])-     np.mean(H1215_k5[:,6])-     np.mean(H1215_k5[:,7])-    np.mean(H1215_k5[:,8])-     np.mean(H1215_k5[:,9])-     np.mean(H1215_k5[:,10])-    np.mean(H1215_k5[:,11])-    np.mean(H1215_k5[:,12]),
                           np.mean(H1215_k4[:,4])-  np.mean(H1215_k4[:,5])-     np.mean(H1215_k4[:,6])-     np.mean(H1215_k4[:,7])-    np.mean(H1215_k4[:,8])-     np.mean(H1215_k4[:,9])-     np.mean(H1215_k4[:,10])-    np.mean(H1215_k4[:,11])-    np.mean(H1215_k4[:,12])   ])
H_overlap2Means.append([   np.mean(H1215_k5[:,5]),  np.mean(H1215_k7[:,5]),     np.mean(H1215_k6[:,5]),     np.mean(H1215_k5[:,5]),    np.mean(H1215_k4[:,5])  ])
H_overlap3Means.append([   np.mean(H1215_k5[:,6]),  np.mean(H1215_k7[:,6]),     np.mean(H1215_k6[:,6]),     np.mean(H1215_k5[:,6]),    np.mean(H1215_k4[:,6])  ])
H_overlap4Means.append([   np.mean(H1215_k5[:,7]),  np.mean(H1215_k7[:,7]),     np.mean(H1215_k6[:,7]),     np.mean(H1215_k5[:,7]),    np.mean(H1215_k4[:,7])  ])
H_overlap5Means.append([   np.mean(H1215_k5[:,8]),  np.mean(H1215_k7[:,8]),     np.mean(H1215_k6[:,8]),     np.mean(H1215_k5[:,8]),    np.mean(H1215_k4[:,8])  ])
H_overlap6Means.append([   np.mean(H1215_k5[:,9]),  np.mean(H1215_k7[:,9]),     np.mean(H1215_k6[:,9]),     np.mean(H1215_k5[:,9]),    np.mean(H1215_k4[:,9])  ])
H_overlap7Means.append([   np.mean(H1215_k5[:,10]), np.mean(H1215_k7[:,10]),    np.mean(H1215_k6[:,10]),    np.mean(H1215_k5[:,10]),   np.mean(H1215_k4[:,10]) ])
H_overlap8Means.append([   np.mean(H1215_k5[:,11]), np.mean(H1215_k7[:,11]),    np.mean(H1215_k6[:,11]),    np.mean(H1215_k5[:,11]),   np.mean(H1215_k4[:,11]) ])
H_overlap9Means.append([   np.mean(H1215_k5[:,12]), np.mean(H1215_k7[:,12]),    np.mean(H1215_k6[:,12]),    np.mean(H1215_k5[:,12]),   np.mean(H1215_k4[:,12]) ])
H_semOverlapStd.append([   np.std(H1215_k5[:,4])-   np.std(H1215_k5[:,5])-      np.std(H1215_k5[:,6])-      np.std(H1215_k5[:,7])-     np.std(H1215_k5[:,8])-     np.std(H1215_k5[:,9])-     np.std(H1215_k5[:,10])-    np.std(H1215_k5[:,11])-    np.std(H1215_k5[:,12]),
                           np.std(H1215_k7[:,4])-   np.std(H1215_k7[:,5])-      np.std(H1215_k7[:,6])-      np.std(H1215_k7[:,7])-     np.std(H1215_k7[:,8])-     np.std(H1215_k7[:,9])-     np.std(H1215_k7[:,10])-    np.std(H1215_k7[:,11])-    np.std(H1215_k7[:,12]),
                           np.std(H1215_k6[:,4])-   np.std(H1215_k6[:,5])-      np.std(H1215_k6[:,6])-      np.std(H1215_k6[:,7])-     np.std(H1215_k6[:,8])-     np.std(H1215_k6[:,9])-     np.std(H1215_k6[:,10])-    np.std(H1215_k6[:,11])-    np.std(H1215_k6[:,12]),
                           np.std(H1215_k5[:,4])-   np.std(H1215_k5[:,5])-      np.std(H1215_k5[:,6])-      np.std(H1215_k5[:,7])-     np.std(H1215_k5[:,8])-     np.std(H1215_k5[:,9])-     np.std(H1215_k5[:,10])-    np.std(H1215_k5[:,11])-    np.std(H1215_k5[:,12]),
                           np.std(H1215_k4[:,4])-   np.std(H1215_k4[:,5])-      np.std(H1215_k4[:,6])-      np.std(H1215_k4[:,7])-     np.std(H1215_k4[:,8])-     np.std(H1215_k4[:,9])-     np.std(H1215_k4[:,10])-    np.std(H1215_k4[:,11])-    np.std(H1215_k4[:,12])   ])
H_overlap2Std.append([     np.std(H1215_k5[:,5]),   np.std(H1215_k7[:,5]),      np.std(H1215_k6[:,5]),      np.std(H1215_k5[:,5]),     np.std(H1215_k4[:,5])  ])
H_overlap3Std.append([     np.std(H1215_k5[:,6]),   np.std(H1215_k7[:,6]),      np.std(H1215_k6[:,6]),      np.std(H1215_k5[:,6]),     np.std(H1215_k4[:,6])  ])
H_overlap4Std.append([     np.std(H1215_k5[:,7]),   np.std(H1215_k7[:,7]),      np.std(H1215_k6[:,7]),      np.std(H1215_k5[:,7]),     np.std(H1215_k4[:,7])  ])
H_overlap5Std.append([     np.std(H1215_k5[:,8]),   np.std(H1215_k7[:,8]),      np.std(H1215_k6[:,8]),      np.std(H1215_k5[:,8]),     np.std(H1215_k4[:,8])  ])
H_overlap6Std.append([     np.std(H1215_k5[:,9]),   np.std(H1215_k7[:,9]),      np.std(H1215_k6[:,9]),      np.std(H1215_k5[:,9]),     np.std(H1215_k4[:,9])  ])
H_overlap7Std.append([     np.std(H1215_k5[:,10]),  np.std(H1215_k7[:,10]),     np.std(H1215_k6[:,10]),     np.std(H1215_k5[:,10]),    np.std(H1215_k4[:,10]) ])
H_overlap8Std.append([     np.std(H1215_k5[:,11]),  np.std(H1215_k7[:,11]),     np.std(H1215_k6[:,11]),     np.std(H1215_k5[:,11]),    np.std(H1215_k4[:,11]) ])
H_overlap9Std.append([     np.std(H1215_k5[:,12]),  np.std(H1215_k7[:,12]),     np.std(H1215_k6[:,12]),     np.std(H1215_k5[:,12]),    np.std(H1215_k4[:,12]) ])



###   Cenário Brum2025   ###
E_semOverlapMeans.append([ E2025_i3c08[4]-  E2025_i3c08[5]-     E2025_i3c08[6]-     E2025_i3c08[7]-     E2025_i3c08[8]-     E2025_i3c08[9]-     E2025_i3c08[10]-    E2025_i3c08[11]-    E2025_i3c08[12],
                           E2025_i3c09[4]-  E2025_i3c09[5]-     E2025_i3c09[6]-     E2025_i3c09[7]-     E2025_i3c09[8]-     E2025_i3c09[9]-     E2025_i3c09[10]-    E2025_i3c09[11]-    E2025_i3c09[12],
                           E2025_i3c095[4]- E2025_i3c095[5]-    E2025_i3c095[6]-    E2025_i3c095[7]-    E2025_i3c095[8]-    E2025_i3c095[9]-    E2025_i3c095[10]-   E2025_i3c095[11]-   E2025_i3c095[12],
                           E2025_i3c098[4]- E2025_i3c098[5]-    E2025_i3c098[6]-    E2025_i3c098[7]-    E2025_i3c098[8]-    E2025_i3c098[9]-    E2025_i3c098[10]-   E2025_i3c098[11]-   E2025_i3c098[12],
                           E2025_i3c1[4]-   E2025_i3c1[5]-      E2025_i3c1[6]-      E2025_i3c1[7]-      E2025_i3c1[8]-      E2025_i3c1[9]-      E2025_i3c1[10]-     E2025_i3c1[11]-     E2025_i3c1[12]   ])
E_overlap2Means.append([   E2025_i3c08[5],  E2025_i3c09[5],     E2025_i3c095[5],    E2025_i3c098[5],    E2025_i3c1[5]  ])
E_overlap3Means.append([   E2025_i3c08[6],  E2025_i3c09[6],     E2025_i3c095[6],    E2025_i3c098[6],    E2025_i3c1[6]  ])
E_overlap4Means.append([   E2025_i3c08[7],  E2025_i3c09[7],     E2025_i3c095[7],    E2025_i3c098[7],    E2025_i3c1[7]  ])
E_overlap5Means.append([   E2025_i3c08[8],  E2025_i3c09[8],     E2025_i3c095[8],    E2025_i3c098[8],    E2025_i3c1[8]  ])
E_overlap6Means.append([   E2025_i3c08[9],  E2025_i3c09[9],     E2025_i3c095[9],    E2025_i3c098[9],    E2025_i3c1[9]  ])
E_overlap7Means.append([   E2025_i3c08[10], E2025_i3c09[10],    E2025_i3c095[10],   E2025_i3c098[10],   E2025_i3c1[10] ])
E_overlap8Means.append([   E2025_i3c08[11], E2025_i3c09[11],    E2025_i3c095[11],   E2025_i3c098[11],   E2025_i3c1[11] ])
E_overlap9Means.append([   E2025_i3c08[12], E2025_i3c09[12],    E2025_i3c095[12],   E2025_i3c098[12],   E2025_i3c1[12] ])

M_semOverlapMeans.append([ np.mean(M2025_i3c08[:,4])-  np.mean(M2025_i3c08[:,5])-     np.mean(M2025_i3c08[:,6])-     np.mean(M2025_i3c08[:,7])-     np.mean(M2025_i3c08[:,8])-     np.mean(M2025_i3c08[:,9])-     np.mean(M2025_i3c08[:,10])-    np.mean(M2025_i3c08[:,11])-    np.mean(M2025_i3c08[:,12]),
                           np.mean(M2025_i3c09[:,4])-  np.mean(M2025_i3c09[:,5])-     np.mean(M2025_i3c09[:,6])-     np.mean(M2025_i3c09[:,7])-     np.mean(M2025_i3c09[:,8])-     np.mean(M2025_i3c09[:,9])-     np.mean(M2025_i3c09[:,10])-    np.mean(M2025_i3c09[:,11])-    np.mean(M2025_i3c09[:,12]),
                           np.mean(M2025_i3c095[:,4])- np.mean(M2025_i3c095[:,5])-    np.mean(M2025_i3c095[:,6])-    np.mean(M2025_i3c095[:,7])-    np.mean(M2025_i3c095[:,8])-    np.mean(M2025_i3c095[:,9])-    np.mean(M2025_i3c095[:,10])-   np.mean(M2025_i3c095[:,11])-   np.mean(M2025_i3c095[:,12]),
                           np.mean(M2025_i3c098[:,4])- np.mean(M2025_i3c098[:,5])-    np.mean(M2025_i3c098[:,6])-    np.mean(M2025_i3c098[:,7])-    np.mean(M2025_i3c098[:,8])-    np.mean(M2025_i3c098[:,9])-    np.mean(M2025_i3c098[:,10])-   np.mean(M2025_i3c098[:,11])-   np.mean(M2025_i3c098[:,12]),
                           np.mean(M2025_i3c1[:,4])-   np.mean(M2025_i3c1[:,5])-      np.mean(M2025_i3c1[:,6])-      np.mean(M2025_i3c1[:,7])-      np.mean(M2025_i3c1[:,8])-      np.mean(M2025_i3c1[:,9])-      np.mean(M2025_i3c1[:,10])-     np.mean(M2025_i3c1[:,11])-     np.mean(M2025_i3c1[:,12])   ])
M_overlap2Means.append([   np.mean(M2025_i3c08[:,5]),  np.mean(M2025_i3c09[:,5]),     np.mean(M2025_i3c095[:,5]),    np.mean(M2025_i3c098[:,5]),    np.mean(M2025_i3c1[:,5])  ])
M_overlap3Means.append([   np.mean(M2025_i3c08[:,6]),  np.mean(M2025_i3c09[:,6]),     np.mean(M2025_i3c095[:,6]),    np.mean(M2025_i3c098[:,6]),    np.mean(M2025_i3c1[:,6])  ])
M_overlap4Means.append([   np.mean(M2025_i3c08[:,7]),  np.mean(M2025_i3c09[:,7]),     np.mean(M2025_i3c095[:,7]),    np.mean(M2025_i3c098[:,7]),    np.mean(M2025_i3c1[:,7])  ])
M_overlap5Means.append([   np.mean(M2025_i3c08[:,8]),  np.mean(M2025_i3c09[:,8]),     np.mean(M2025_i3c095[:,8]),    np.mean(M2025_i3c098[:,8]),    np.mean(M2025_i3c1[:,8])  ])
M_overlap6Means.append([   np.mean(M2025_i3c08[:,9]),  np.mean(M2025_i3c09[:,9]),     np.mean(M2025_i3c095[:,9]),    np.mean(M2025_i3c098[:,9]),    np.mean(M2025_i3c1[:,9])  ])
M_overlap7Means.append([   np.mean(M2025_i3c08[:,10]), np.mean(M2025_i3c09[:,10]),    np.mean(M2025_i3c095[:,10]),   np.mean(M2025_i3c098[:,10]),   np.mean(M2025_i3c1[:,10]) ])
M_overlap8Means.append([   np.mean(M2025_i3c08[:,11]), np.mean(M2025_i3c09[:,11]),    np.mean(M2025_i3c095[:,11]),   np.mean(M2025_i3c098[:,11]),   np.mean(M2025_i3c1[:,11]) ])
M_overlap9Means.append([   np.mean(M2025_i3c08[:,12]), np.mean(M2025_i3c09[:,12]),    np.mean(M2025_i3c095[:,12]),   np.mean(M2025_i3c098[:,12]),   np.mean(M2025_i3c1[:,12]) ])
M_semOverlapStd.append([   np.std(M2025_i3c08[:,4])-   np.std(M2025_i3c08[:,5])-      np.std(M2025_i3c08[:,6])-      np.std(M2025_i3c08[:,7])-      np.std(M2025_i3c08[:,8])-     np.std(M2025_i3c08[:,9])-     np.std(M2025_i3c08[:,10])-    np.std(M2025_i3c08[:,11])-    np.std(M2025_i3c08[:,12]),
                           np.std(M2025_i3c09[:,4])-   np.std(M2025_i3c09[:,5])-      np.std(M2025_i3c09[:,6])-      np.std(M2025_i3c09[:,7])-      np.std(M2025_i3c09[:,8])-     np.std(M2025_i3c09[:,9])-     np.std(M2025_i3c09[:,10])-    np.std(M2025_i3c09[:,11])-    np.std(M2025_i3c09[:,12]),
                           np.std(M2025_i3c095[:,4])-  np.std(M2025_i3c095[:,5])-     np.std(M2025_i3c095[:,6])-     np.std(M2025_i3c095[:,7])-     np.std(M2025_i3c095[:,8])-    np.std(M2025_i3c095[:,9])-    np.std(M2025_i3c095[:,10])-   np.std(M2025_i3c095[:,11])-   np.std(M2025_i3c095[:,12]),
                           np.std(M2025_i3c098[:,4])-  np.std(M2025_i3c098[:,5])-     np.std(M2025_i3c098[:,6])-     np.std(M2025_i3c098[:,7])-     np.std(M2025_i3c098[:,8])-    np.std(M2025_i3c098[:,9])-    np.std(M2025_i3c098[:,10])-   np.std(M2025_i3c098[:,11])-   np.std(M2025_i3c098[:,12]),
                           np.std(M2025_i3c1[:,4])-    np.std(M2025_i3c1[:,5])-       np.std(M2025_i3c1[:,6])-       np.std(M2025_i3c1[:,7])-       np.std(M2025_i3c1[:,8])-      np.std(M2025_i3c1[:,9])-      np.std(M2025_i3c1[:,10])-     np.std(M2025_i3c1[:,11])-     np.std(M2025_i3c1[:,12])   ])
M_overlap2Std.append([     np.std(M2025_i3c08[:,5]),   np.std(M2025_i3c09[:,5]),      np.std(M2025_i3c095[:,5]),     np.std(M2025_i3c098[:,5]),     np.std(M2025_i3c1[:,5])  ])
M_overlap3Std.append([     np.std(M2025_i3c08[:,6]),   np.std(M2025_i3c09[:,6]),      np.std(M2025_i3c095[:,6]),     np.std(M2025_i3c098[:,6]),     np.std(M2025_i3c1[:,6])  ])
M_overlap4Std.append([     np.std(M2025_i3c08[:,7]),   np.std(M2025_i3c09[:,7]),      np.std(M2025_i3c095[:,7]),     np.std(M2025_i3c098[:,7]),     np.std(M2025_i3c1[:,7])  ])
M_overlap5Std.append([     np.std(M2025_i3c08[:,8]),   np.std(M2025_i3c09[:,8]),      np.std(M2025_i3c095[:,8]),     np.std(M2025_i3c098[:,8]),     np.std(M2025_i3c1[:,8])  ])
M_overlap6Std.append([     np.std(M2025_i3c08[:,9]),   np.std(M2025_i3c09[:,9]),      np.std(M2025_i3c095[:,9]),     np.std(M2025_i3c098[:,9]),     np.std(M2025_i3c1[:,9])  ])
M_overlap7Std.append([     np.std(M2025_i3c08[:,10]),  np.std(M2025_i3c09[:,10]),     np.std(M2025_i3c095[:,10]),    np.std(M2025_i3c098[:,10]),    np.std(M2025_i3c1[:,10]) ])
M_overlap8Std.append([     np.std(M2025_i3c08[:,11]),  np.std(M2025_i3c09[:,11]),     np.std(M2025_i3c095[:,11]),    np.std(M2025_i3c098[:,11]),    np.std(M2025_i3c1[:,11]) ])
M_overlap9Std.append([     np.std(M2025_i3c08[:,12]),  np.std(M2025_i3c09[:,12]),     np.std(M2025_i3c095[:,12]),    np.std(M2025_i3c098[:,12]),    np.std(M2025_i3c1[:,12]) ])

H_semOverlapMeans.append([ np.mean(H2025_k6[:,4])-  np.mean(H2025_k6[:,5])-     np.mean(H2025_k6[:,6])-     np.mean(H2025_k6[:,7])-    np.mean(H2025_k6[:,8])-     np.mean(H2025_k6[:,9])-     np.mean(H2025_k6[:,10])-    np.mean(H2025_k6[:,11])-    np.mean(H2025_k6[:,12]),
                           np.mean(H2025_k6[:,4])-  np.mean(H2025_k6[:,5])-     np.mean(H2025_k6[:,6])-     np.mean(H2025_k6[:,7])-    np.mean(H2025_k6[:,8])-     np.mean(H2025_k6[:,9])-     np.mean(H2025_k6[:,10])-    np.mean(H2025_k6[:,11])-    np.mean(H2025_k6[:,12]),
                           np.mean(H2025_k7[:,4])-  np.mean(H2025_k7[:,5])-     np.mean(H2025_k7[:,6])-     np.mean(H2025_k7[:,7])-    np.mean(H2025_k7[:,8])-     np.mean(H2025_k7[:,9])-     np.mean(H2025_k7[:,10])-    np.mean(H2025_k7[:,11])-    np.mean(H2025_k7[:,12]),
                           np.mean(H2025_k7[:,4])-  np.mean(H2025_k7[:,5])-     np.mean(H2025_k7[:,6])-     np.mean(H2025_k7[:,7])-    np.mean(H2025_k7[:,8])-     np.mean(H2025_k7[:,9])-     np.mean(H2025_k7[:,10])-    np.mean(H2025_k7[:,11])-    np.mean(H2025_k7[:,12]),
                           np.mean(H2025_k6[:,4])-  np.mean(H2025_k6[:,5])-     np.mean(H2025_k6[:,6])-     np.mean(H2025_k6[:,7])-    np.mean(H2025_k6[:,8])-     np.mean(H2025_k6[:,9])-     np.mean(H2025_k6[:,10])-    np.mean(H2025_k6[:,11])-    np.mean(H2025_k6[:,12])   ])
H_overlap2Means.append([   np.mean(H2025_k6[:,5]),  np.mean(H2025_k6[:,5]),     np.mean(H2025_k7[:,5]),     np.mean(H2025_k7[:,5]),    np.mean(H2025_k6[:,5])  ])
H_overlap3Means.append([   np.mean(H2025_k6[:,6]),  np.mean(H2025_k6[:,6]),     np.mean(H2025_k7[:,6]),     np.mean(H2025_k7[:,6]),    np.mean(H2025_k6[:,6])  ])
H_overlap4Means.append([   np.mean(H2025_k6[:,7]),  np.mean(H2025_k6[:,7]),     np.mean(H2025_k7[:,7]),     np.mean(H2025_k7[:,7]),    np.mean(H2025_k6[:,7])  ])
H_overlap5Means.append([   np.mean(H2025_k6[:,8]),  np.mean(H2025_k6[:,8]),     np.mean(H2025_k7[:,8]),     np.mean(H2025_k7[:,8]),    np.mean(H2025_k6[:,8])  ])
H_overlap6Means.append([   np.mean(H2025_k6[:,9]),  np.mean(H2025_k6[:,9]),     np.mean(H2025_k7[:,9]),     np.mean(H2025_k7[:,9]),    np.mean(H2025_k6[:,9])  ])
H_overlap7Means.append([   np.mean(H2025_k6[:,10]), np.mean(H2025_k6[:,10]),    np.mean(H2025_k7[:,10]),    np.mean(H2025_k7[:,10]),   np.mean(H2025_k6[:,10]) ])
H_overlap8Means.append([   np.mean(H2025_k6[:,11]), np.mean(H2025_k6[:,11]),    np.mean(H2025_k7[:,11]),    np.mean(H2025_k7[:,11]),   np.mean(H2025_k6[:,11]) ])
H_overlap9Means.append([   np.mean(H2025_k6[:,12]), np.mean(H2025_k6[:,12]),    np.mean(H2025_k7[:,12]),    np.mean(H2025_k7[:,12]),   np.mean(H2025_k6[:,12]) ])
H_semOverlapStd.append([   np.std(H2025_k6[:,4])-   np.std(H2025_k6[:,5])-      np.std(H2025_k6[:,6])-      np.std(H2025_k6[:,7])-     np.std(H2025_k6[:,8])-     np.std(H2025_k6[:,9])-     np.std(H2025_k6[:,10])-    np.std(H2025_k6[:,11])-    np.std(H2025_k6[:,12]),
                           np.std(H2025_k6[:,4])-   np.std(H2025_k6[:,5])-      np.std(H2025_k6[:,6])-      np.std(H2025_k6[:,7])-     np.std(H2025_k6[:,8])-     np.std(H2025_k6[:,9])-     np.std(H2025_k6[:,10])-    np.std(H2025_k6[:,11])-    np.std(H2025_k6[:,12]),
                           np.std(H2025_k7[:,4])-   np.std(H2025_k7[:,5])-      np.std(H2025_k7[:,6])-      np.std(H2025_k7[:,7])-     np.std(H2025_k7[:,8])-     np.std(H2025_k7[:,9])-     np.std(H2025_k7[:,10])-    np.std(H2025_k7[:,11])-    np.std(H2025_k7[:,12]),
                           np.std(H2025_k7[:,4])-   np.std(H2025_k7[:,5])-      np.std(H2025_k7[:,6])-      np.std(H2025_k7[:,7])-     np.std(H2025_k7[:,8])-     np.std(H2025_k7[:,9])-     np.std(H2025_k7[:,10])-    np.std(H2025_k7[:,11])-    np.std(H2025_k7[:,12]),
                           np.std(H2025_k6[:,4])-   np.std(H2025_k6[:,5])-      np.std(H2025_k6[:,6])-      np.std(H2025_k6[:,7])-     np.std(H2025_k6[:,8])-     np.std(H2025_k6[:,9])-     np.std(H2025_k6[:,10])-    np.std(H2025_k6[:,11])-    np.std(H2025_k6[:,12])   ])
H_overlap2Std.append([     np.std(H2025_k6[:,5]),   np.std(H2025_k6[:,5]),      np.std(H2025_k7[:,5]),      np.std(H2025_k7[:,5]),     np.std(H2025_k6[:,5])  ])
H_overlap3Std.append([     np.std(H2025_k6[:,6]),   np.std(H2025_k6[:,6]),      np.std(H2025_k7[:,6]),      np.std(H2025_k7[:,6]),     np.std(H2025_k6[:,6])  ])
H_overlap4Std.append([     np.std(H2025_k6[:,7]),   np.std(H2025_k6[:,7]),      np.std(H2025_k7[:,7]),      np.std(H2025_k7[:,7]),     np.std(H2025_k6[:,7])  ])
H_overlap5Std.append([     np.std(H2025_k6[:,8]),   np.std(H2025_k6[:,8]),      np.std(H2025_k7[:,8]),      np.std(H2025_k7[:,8]),     np.std(H2025_k6[:,8])  ])
H_overlap6Std.append([     np.std(H2025_k6[:,9]),   np.std(H2025_k6[:,9]),      np.std(H2025_k7[:,9]),      np.std(H2025_k7[:,9]),     np.std(H2025_k6[:,9])  ])
H_overlap7Std.append([     np.std(H2025_k6[:,10]),  np.std(H2025_k6[:,10]),     np.std(H2025_k7[:,10]),     np.std(H2025_k7[:,10]),    np.std(H2025_k6[:,10]) ])
H_overlap8Std.append([     np.std(H2025_k6[:,11]),  np.std(H2025_k6[:,11]),     np.std(H2025_k7[:,11]),     np.std(H2025_k7[:,11]),    np.std(H2025_k6[:,11]) ])
H_overlap9Std.append([     np.std(H2025_k6[:,12]),  np.std(H2025_k6[:,12]),     np.std(H2025_k7[:,12]),     np.std(H2025_k7[:,12]),    np.std(H2025_k6[:,12]) ])



###   Cenário Brum2430   ###
E_semOverlapMeans.append([ E2430_i3c08[4]-  E2430_i3c08[5]-     E2430_i3c08[6]-     E2430_i3c08[7]-     E2430_i3c08[8]-     E2430_i3c08[9]-     E2430_i3c08[10]-    E2430_i3c08[11]-    E2430_i3c08[12],
                           E2430_i3c09[4]-  E2430_i3c09[5]-     E2430_i3c09[6]-     E2430_i3c09[7]-     E2430_i3c09[8]-     E2430_i3c09[9]-     E2430_i3c09[10]-    E2430_i3c09[11]-    E2430_i3c09[12],
                           E2430_i3c095[4]- E2430_i3c095[5]-    E2430_i3c095[6]-    E2430_i3c095[7]-    E2430_i3c095[8]-    E2430_i3c095[9]-    E2430_i3c095[10]-   E2430_i3c095[11]-   E2430_i3c095[12],
                           E2430_i3c098[4]- E2430_i3c098[5]-    E2430_i3c098[6]-    E2430_i3c098[7]-    E2430_i3c098[8]-    E2430_i3c098[9]-    E2430_i3c098[10]-   E2430_i3c098[11]-   E2430_i3c098[12],
                           E2430_i3c1[4]-   E2430_i3c1[5]-      E2430_i3c1[6]-      E2430_i3c1[7]-      E2430_i3c1[8]-      E2430_i3c1[9]-      E2430_i3c1[10]-     E2430_i3c1[11]-     E2430_i3c1[12]   ])
E_overlap2Means.append([   E2430_i3c08[5],  E2430_i3c09[5],     E2430_i3c095[5],    E2430_i3c098[5],    E2430_i3c1[5]  ])
E_overlap3Means.append([   E2430_i3c08[6],  E2430_i3c09[6],     E2430_i3c095[6],    E2430_i3c098[6],    E2430_i3c1[6]  ])
E_overlap4Means.append([   E2430_i3c08[7],  E2430_i3c09[7],     E2430_i3c095[7],    E2430_i3c098[7],    E2430_i3c1[7]  ])
E_overlap5Means.append([   E2430_i3c08[8],  E2430_i3c09[8],     E2430_i3c095[8],    E2430_i3c098[8],    E2430_i3c1[8]  ])
E_overlap6Means.append([   E2430_i3c08[9],  E2430_i3c09[9],     E2430_i3c095[9],    E2430_i3c098[9],    E2430_i3c1[9]  ])
E_overlap7Means.append([   E2430_i3c08[10], E2430_i3c09[10],    E2430_i3c095[10],   E2430_i3c098[10],   E2430_i3c1[10] ])
E_overlap8Means.append([   E2430_i3c08[11], E2430_i3c09[11],    E2430_i3c095[11],   E2430_i3c098[11],   E2430_i3c1[11] ])
E_overlap9Means.append([   E2430_i3c08[12], E2430_i3c09[12],    E2430_i3c095[12],   E2430_i3c098[12],   E2430_i3c1[12] ])

M_semOverlapMeans.append([ np.mean(M2430_i3c08[:,4])-  np.mean(M2430_i3c08[:,5])-     np.mean(M2430_i3c08[:,6])-     np.mean(M2430_i3c08[:,7])-     np.mean(M2430_i3c08[:,8])-     np.mean(M2430_i3c08[:,9])-     np.mean(M2430_i3c08[:,10])-    np.mean(M2430_i3c08[:,11])-    np.mean(M2430_i3c08[:,12]),
                           np.mean(M2430_i3c09[:,4])-  np.mean(M2430_i3c09[:,5])-     np.mean(M2430_i3c09[:,6])-     np.mean(M2430_i3c09[:,7])-     np.mean(M2430_i3c09[:,8])-     np.mean(M2430_i3c09[:,9])-     np.mean(M2430_i3c09[:,10])-    np.mean(M2430_i3c09[:,11])-    np.mean(M2430_i3c09[:,12]),
                           np.mean(M2430_i3c095[:,4])- np.mean(M2430_i3c095[:,5])-    np.mean(M2430_i3c095[:,6])-    np.mean(M2430_i3c095[:,7])-    np.mean(M2430_i3c095[:,8])-    np.mean(M2430_i3c095[:,9])-    np.mean(M2430_i3c095[:,10])-   np.mean(M2430_i3c095[:,11])-   np.mean(M2430_i3c095[:,12]),
                           np.mean(M2430_i3c098[:,4])- np.mean(M2430_i3c098[:,5])-    np.mean(M2430_i3c098[:,6])-    np.mean(M2430_i3c098[:,7])-    np.mean(M2430_i3c098[:,8])-    np.mean(M2430_i3c098[:,9])-    np.mean(M2430_i3c098[:,10])-   np.mean(M2430_i3c098[:,11])-   np.mean(M2430_i3c098[:,12]),
                           np.mean(M2430_i3c1[:,4])-   np.mean(M2430_i3c1[:,5])-      np.mean(M2430_i3c1[:,6])-      np.mean(M2430_i3c1[:,7])-      np.mean(M2430_i3c1[:,8])-      np.mean(M2430_i3c1[:,9])-      np.mean(M2430_i3c1[:,10])-     np.mean(M2430_i3c1[:,11])-     np.mean(M2430_i3c1[:,12])   ])
M_overlap2Means.append([   np.mean(M2430_i3c08[:,5]),  np.mean(M2430_i3c09[:,5]),     np.mean(M2430_i3c095[:,5]),    np.mean(M2430_i3c098[:,5]),    np.mean(M2430_i3c1[:,5])  ])
M_overlap3Means.append([   np.mean(M2430_i3c08[:,6]),  np.mean(M2430_i3c09[:,6]),     np.mean(M2430_i3c095[:,6]),    np.mean(M2430_i3c098[:,6]),    np.mean(M2430_i3c1[:,6])  ])
M_overlap4Means.append([   np.mean(M2430_i3c08[:,7]),  np.mean(M2430_i3c09[:,7]),     np.mean(M2430_i3c095[:,7]),    np.mean(M2430_i3c098[:,7]),    np.mean(M2430_i3c1[:,7])  ])
M_overlap5Means.append([   np.mean(M2430_i3c08[:,8]),  np.mean(M2430_i3c09[:,8]),     np.mean(M2430_i3c095[:,8]),    np.mean(M2430_i3c098[:,8]),    np.mean(M2430_i3c1[:,8])  ])
M_overlap6Means.append([   np.mean(M2430_i3c08[:,9]),  np.mean(M2430_i3c09[:,9]),     np.mean(M2430_i3c095[:,9]),    np.mean(M2430_i3c098[:,9]),    np.mean(M2430_i3c1[:,9])  ])
M_overlap7Means.append([   np.mean(M2430_i3c08[:,10]), np.mean(M2430_i3c09[:,10]),    np.mean(M2430_i3c095[:,10]),   np.mean(M2430_i3c098[:,10]),   np.mean(M2430_i3c1[:,10]) ])
M_overlap8Means.append([   np.mean(M2430_i3c08[:,11]), np.mean(M2430_i3c09[:,11]),    np.mean(M2430_i3c095[:,11]),   np.mean(M2430_i3c098[:,11]),   np.mean(M2430_i3c1[:,11]) ])
M_overlap9Means.append([   np.mean(M2430_i3c08[:,12]), np.mean(M2430_i3c09[:,12]),    np.mean(M2430_i3c095[:,12]),   np.mean(M2430_i3c098[:,12]),   np.mean(M2430_i3c1[:,12]) ])
M_semOverlapStd.append([   np.std(M2430_i3c08[:,4])-   np.std(M2430_i3c08[:,5])-      np.std(M2430_i3c08[:,6])-      np.std(M2430_i3c08[:,7])-      np.std(M2430_i3c08[:,8])-     np.std(M2430_i3c08[:,9])-     np.std(M2430_i3c08[:,10])-    np.std(M2430_i3c08[:,11])-    np.std(M2430_i3c08[:,12]),
                           np.std(M2430_i3c09[:,4])-   np.std(M2430_i3c09[:,5])-      np.std(M2430_i3c09[:,6])-      np.std(M2430_i3c09[:,7])-      np.std(M2430_i3c09[:,8])-     np.std(M2430_i3c09[:,9])-     np.std(M2430_i3c09[:,10])-    np.std(M2430_i3c09[:,11])-    np.std(M2430_i3c09[:,12]),
                           np.std(M2430_i3c095[:,4])-  np.std(M2430_i3c095[:,5])-     np.std(M2430_i3c095[:,6])-     np.std(M2430_i3c095[:,7])-     np.std(M2430_i3c095[:,8])-    np.std(M2430_i3c095[:,9])-    np.std(M2430_i3c095[:,10])-   np.std(M2430_i3c095[:,11])-   np.std(M2430_i3c095[:,12]),
                           np.std(M2430_i3c098[:,4])-  np.std(M2430_i3c098[:,5])-     np.std(M2430_i3c098[:,6])-     np.std(M2430_i3c098[:,7])-     np.std(M2430_i3c098[:,8])-    np.std(M2430_i3c098[:,9])-    np.std(M2430_i3c098[:,10])-   np.std(M2430_i3c098[:,11])-   np.std(M2430_i3c098[:,12]),
                           np.std(M2430_i3c1[:,4])-    np.std(M2430_i3c1[:,5])-       np.std(M2430_i3c1[:,6])-       np.std(M2430_i3c1[:,7])-       np.std(M2430_i3c1[:,8])-      np.std(M2430_i3c1[:,9])-      np.std(M2430_i3c1[:,10])-     np.std(M2430_i3c1[:,11])-     np.std(M2430_i3c1[:,12])   ])
M_overlap2Std.append([     np.std(M2430_i3c08[:,5]),   np.std(M2430_i3c09[:,5]),      np.std(M2430_i3c095[:,5]),     np.std(M2430_i3c098[:,5]),     np.std(M2430_i3c1[:,5])  ])
M_overlap3Std.append([     np.std(M2430_i3c08[:,6]),   np.std(M2430_i3c09[:,6]),      np.std(M2430_i3c095[:,6]),     np.std(M2430_i3c098[:,6]),     np.std(M2430_i3c1[:,6])  ])
M_overlap4Std.append([     np.std(M2430_i3c08[:,7]),   np.std(M2430_i3c09[:,7]),      np.std(M2430_i3c095[:,7]),     np.std(M2430_i3c098[:,7]),     np.std(M2430_i3c1[:,7])  ])
M_overlap5Std.append([     np.std(M2430_i3c08[:,8]),   np.std(M2430_i3c09[:,8]),      np.std(M2430_i3c095[:,8]),     np.std(M2430_i3c098[:,8]),     np.std(M2430_i3c1[:,8])  ])
M_overlap6Std.append([     np.std(M2430_i3c08[:,9]),   np.std(M2430_i3c09[:,9]),      np.std(M2430_i3c095[:,9]),     np.std(M2430_i3c098[:,9]),     np.std(M2430_i3c1[:,9])  ])
M_overlap7Std.append([     np.std(M2430_i3c08[:,10]),  np.std(M2430_i3c09[:,10]),     np.std(M2430_i3c095[:,10]),    np.std(M2430_i3c098[:,10]),    np.std(M2430_i3c1[:,10]) ])
M_overlap8Std.append([     np.std(M2430_i3c08[:,11]),  np.std(M2430_i3c09[:,11]),     np.std(M2430_i3c095[:,11]),    np.std(M2430_i3c098[:,11]),    np.std(M2430_i3c1[:,11]) ])
M_overlap9Std.append([     np.std(M2430_i3c08[:,12]),  np.std(M2430_i3c09[:,12]),     np.std(M2430_i3c095[:,12]),    np.std(M2430_i3c098[:,12]),    np.std(M2430_i3c1[:,12]) ])

H_semOverlapMeans.append([ np.mean(H2430_k8[:,4])-  np.mean(H2430_k8[:,5])-     np.mean(H2430_k8[:,6])-     np.mean(H2430_k8[:,7])-    np.mean(H2430_k8[:,8])-     np.mean(H2430_k8[:,9])-     np.mean(H2430_k8[:,10])-    np.mean(H2430_k8[:,11])-    np.mean(H2430_k8[:,12]),
                           np.mean(H2430_k5[:,4])-  np.mean(H2430_k5[:,5])-     np.mean(H2430_k5[:,6])-     np.mean(H2430_k5[:,7])-    np.mean(H2430_k5[:,8])-     np.mean(H2430_k5[:,9])-     np.mean(H2430_k5[:,10])-    np.mean(H2430_k5[:,11])-    np.mean(H2430_k5[:,12]),
                           np.mean(H2430_k6[:,4])-  np.mean(H2430_k6[:,5])-     np.mean(H2430_k6[:,6])-     np.mean(H2430_k6[:,7])-    np.mean(H2430_k6[:,8])-     np.mean(H2430_k6[:,9])-     np.mean(H2430_k6[:,10])-    np.mean(H2430_k6[:,11])-    np.mean(H2430_k6[:,12]),
                           np.mean(H2430_k6[:,4])-  np.mean(H2430_k6[:,5])-     np.mean(H2430_k6[:,6])-     np.mean(H2430_k6[:,7])-    np.mean(H2430_k6[:,8])-     np.mean(H2430_k6[:,9])-     np.mean(H2430_k6[:,10])-    np.mean(H2430_k6[:,11])-    np.mean(H2430_k6[:,12]),
                           np.mean(H2430_k6[:,4])-  np.mean(H2430_k6[:,5])-     np.mean(H2430_k6[:,6])-     np.mean(H2430_k6[:,7])-    np.mean(H2430_k6[:,8])-     np.mean(H2430_k6[:,9])-     np.mean(H2430_k6[:,10])-    np.mean(H2430_k6[:,11])-    np.mean(H2430_k6[:,12])   ])
H_overlap2Means.append([   np.mean(H2430_k8[:,5]),  np.mean(H2430_k5[:,5]),     np.mean(H2430_k6[:,5]),     np.mean(H2430_k6[:,5]),    np.mean(H2430_k6[:,5])  ])
H_overlap3Means.append([   np.mean(H2430_k8[:,6]),  np.mean(H2430_k5[:,6]),     np.mean(H2430_k6[:,6]),     np.mean(H2430_k6[:,6]),    np.mean(H2430_k6[:,6])  ])
H_overlap4Means.append([   np.mean(H2430_k8[:,7]),  np.mean(H2430_k5[:,7]),     np.mean(H2430_k6[:,7]),     np.mean(H2430_k6[:,7]),    np.mean(H2430_k6[:,7])  ])
H_overlap5Means.append([   np.mean(H2430_k8[:,8]),  np.mean(H2430_k5[:,8]),     np.mean(H2430_k6[:,8]),     np.mean(H2430_k6[:,8]),    np.mean(H2430_k6[:,8])  ])
H_overlap6Means.append([   np.mean(H2430_k8[:,9]),  np.mean(H2430_k5[:,9]),     np.mean(H2430_k6[:,9]),     np.mean(H2430_k6[:,9]),    np.mean(H2430_k6[:,9])  ])
H_overlap7Means.append([   np.mean(H2430_k8[:,10]), np.mean(H2430_k5[:,10]),    np.mean(H2430_k6[:,10]),    np.mean(H2430_k6[:,10]),   np.mean(H2430_k6[:,10]) ])
H_overlap8Means.append([   np.mean(H2430_k8[:,11]), np.mean(H2430_k5[:,11]),    np.mean(H2430_k6[:,11]),    np.mean(H2430_k6[:,11]),   np.mean(H2430_k6[:,11]) ])
H_overlap9Means.append([   np.mean(H2430_k8[:,12]), np.mean(H2430_k5[:,12]),    np.mean(H2430_k6[:,12]),    np.mean(H2430_k6[:,12]),   np.mean(H2430_k6[:,12]) ])
H_semOverlapStd.append([   np.std(H2430_k8[:,4])-   np.std(H2430_k8[:,5])-      np.std(H2430_k8[:,6])-      np.std(H2430_k8[:,7])-     np.std(H2430_k8[:,8])-     np.std(H2430_k8[:,9])-     np.std(H2430_k8[:,10])-    np.std(H2430_k8[:,11])-    np.std(H2430_k8[:,12]),
                           np.std(H2430_k5[:,4])-   np.std(H2430_k5[:,5])-      np.std(H2430_k5[:,6])-      np.std(H2430_k5[:,7])-     np.std(H2430_k5[:,8])-     np.std(H2430_k5[:,9])-     np.std(H2430_k5[:,10])-    np.std(H2430_k5[:,11])-    np.std(H2430_k5[:,12]),
                           np.std(H2430_k6[:,4])-   np.std(H2430_k6[:,5])-      np.std(H2430_k6[:,6])-      np.std(H2430_k6[:,7])-     np.std(H2430_k6[:,8])-     np.std(H2430_k6[:,9])-     np.std(H2430_k6[:,10])-    np.std(H2430_k6[:,11])-    np.std(H2430_k6[:,12]),
                           np.std(H2430_k6[:,4])-   np.std(H2430_k6[:,5])-      np.std(H2430_k6[:,6])-      np.std(H2430_k6[:,7])-     np.std(H2430_k6[:,8])-     np.std(H2430_k6[:,9])-     np.std(H2430_k6[:,10])-    np.std(H2430_k6[:,11])-    np.std(H2430_k6[:,12]),
                           np.std(H2430_k6[:,4])-   np.std(H2430_k6[:,5])-      np.std(H2430_k6[:,6])-      np.std(H2430_k6[:,7])-     np.std(H2430_k6[:,8])-     np.std(H2430_k6[:,9])-     np.std(H2430_k6[:,10])-    np.std(H2430_k6[:,11])-    np.std(H2430_k6[:,12])   ])
H_overlap2Std.append([     np.std(H2430_k8[:,5]),   np.std(H2430_k5[:,5]),      np.std(H2430_k6[:,5]),      np.std(H2430_k6[:,5]),     np.std(H2430_k6[:,5])  ])
H_overlap3Std.append([     np.std(H2430_k8[:,6]),   np.std(H2430_k5[:,6]),      np.std(H2430_k6[:,6]),      np.std(H2430_k6[:,6]),     np.std(H2430_k6[:,6])  ])
H_overlap4Std.append([     np.std(H2430_k8[:,7]),   np.std(H2430_k5[:,7]),      np.std(H2430_k6[:,7]),      np.std(H2430_k6[:,7]),     np.std(H2430_k6[:,7])  ])
H_overlap5Std.append([     np.std(H2430_k8[:,8]),   np.std(H2430_k5[:,8]),      np.std(H2430_k6[:,8]),      np.std(H2430_k6[:,8]),     np.std(H2430_k6[:,8])  ])
H_overlap6Std.append([     np.std(H2430_k8[:,9]),   np.std(H2430_k5[:,9]),      np.std(H2430_k6[:,9]),      np.std(H2430_k6[:,9]),     np.std(H2430_k6[:,9])  ])
H_overlap7Std.append([     np.std(H2430_k8[:,10]),  np.std(H2430_k5[:,10]),     np.std(H2430_k6[:,10]),     np.std(H2430_k6[:,10]),    np.std(H2430_k6[:,10]) ])
H_overlap8Std.append([     np.std(H2430_k8[:,11]),  np.std(H2430_k5[:,11]),     np.std(H2430_k6[:,11]),     np.std(H2430_k6[:,11]),    np.std(H2430_k6[:,11]) ])
H_overlap9Std.append([     np.std(H2430_k8[:,12]),  np.std(H2430_k5[:,12]),     np.std(H2430_k6[:,12]),     np.std(H2430_k6[:,12]),    np.std(H2430_k6[:,12]) ])



###   Cenário Brum3645   ###
E_semOverlapMeans.append([ E3645_i3c08[4]-  E3645_i3c08[5]-     E3645_i3c08[6]-     E3645_i3c08[7]-     E3645_i3c08[8]-     E3645_i3c08[9]-     E3645_i3c08[10]-    E3645_i3c08[11]-    E3645_i3c08[12],
                           E3645_i3c09[4]-  E3645_i3c09[5]-     E3645_i3c09[6]-     E3645_i3c09[7]-     E3645_i3c09[8]-     E3645_i3c09[9]-     E3645_i3c09[10]-    E3645_i3c09[11]-    E3645_i3c09[12],
                           E3645_i3c095[4]- E3645_i3c095[5]-    E3645_i3c095[6]-    E3645_i3c095[7]-    E3645_i3c095[8]-    E3645_i3c095[9]-    E3645_i3c095[10]-   E3645_i3c095[11]-   E3645_i3c095[12],
                           E3645_i3c098[4]- E3645_i3c098[5]-    E3645_i3c098[6]-    E3645_i3c098[7]-    E3645_i3c098[8]-    E3645_i3c098[9]-    E3645_i3c098[10]-   E3645_i3c098[11]-   E3645_i3c098[12],
                           E3645_i3c1[4]-   E3645_i3c1[5]-      E3645_i3c1[6]-      E3645_i3c1[7]-      E3645_i3c1[8]-      E3645_i3c1[9]-      E3645_i3c1[10]-     E3645_i3c1[11]-     E3645_i3c1[12]   ])
E_overlap2Means.append([   E3645_i3c08[5],  E3645_i3c09[5],     E3645_i3c095[5],    E3645_i3c098[5],    E3645_i3c1[5]  ])
E_overlap3Means.append([   E3645_i3c08[6],  E3645_i3c09[6],     E3645_i3c095[6],    E3645_i3c098[6],    E3645_i3c1[6]  ])
E_overlap4Means.append([   E3645_i3c08[7],  E3645_i3c09[7],     E3645_i3c095[7],    E3645_i3c098[7],    E3645_i3c1[7]  ])
E_overlap5Means.append([   E3645_i3c08[8],  E3645_i3c09[8],     E3645_i3c095[8],    E3645_i3c098[8],    E3645_i3c1[8]  ])
E_overlap6Means.append([   E3645_i3c08[9],  E3645_i3c09[9],     E3645_i3c095[9],    E3645_i3c098[9],    E3645_i3c1[9]  ])
E_overlap7Means.append([   E3645_i3c08[10], E3645_i3c09[10],    E3645_i3c095[10],   E3645_i3c098[10],   E3645_i3c1[10] ])
E_overlap8Means.append([   E3645_i3c08[11], E3645_i3c09[11],    E3645_i3c095[11],   E3645_i3c098[11],   E3645_i3c1[11] ])
E_overlap9Means.append([   E3645_i3c08[12], E3645_i3c09[12],    E3645_i3c095[12],   E3645_i3c098[12],   E3645_i3c1[12] ])

M_semOverlapMeans.append([ np.mean(M3645_i3c08[:,4])-  np.mean(M3645_i3c08[:,5])-     np.mean(M3645_i3c08[:,6])-     np.mean(M3645_i3c08[:,7])-     np.mean(M3645_i3c08[:,8])-     np.mean(M3645_i3c08[:,9])-     np.mean(M3645_i3c08[:,10])-    np.mean(M3645_i3c08[:,11])-    np.mean(M3645_i3c08[:,12]),
                           np.mean(M3645_i3c09[:,4])-  np.mean(M3645_i3c09[:,5])-     np.mean(M3645_i3c09[:,6])-     np.mean(M3645_i3c09[:,7])-     np.mean(M3645_i3c09[:,8])-     np.mean(M3645_i3c09[:,9])-     np.mean(M3645_i3c09[:,10])-    np.mean(M3645_i3c09[:,11])-    np.mean(M3645_i3c09[:,12]),
                           np.mean(M3645_i3c095[:,4])- np.mean(M3645_i3c095[:,5])-    np.mean(M3645_i3c095[:,6])-    np.mean(M3645_i3c095[:,7])-    np.mean(M3645_i3c095[:,8])-    np.mean(M3645_i3c095[:,9])-    np.mean(M3645_i3c095[:,10])-   np.mean(M3645_i3c095[:,11])-   np.mean(M3645_i3c095[:,12]),
                           np.mean(M3645_i3c098[:,4])- np.mean(M3645_i3c098[:,5])-    np.mean(M3645_i3c098[:,6])-    np.mean(M3645_i3c098[:,7])-    np.mean(M3645_i3c098[:,8])-    np.mean(M3645_i3c098[:,9])-    np.mean(M3645_i3c098[:,10])-   np.mean(M3645_i3c098[:,11])-   np.mean(M3645_i3c098[:,12]),
                           np.mean(M3645_i3c1[:,4])-   np.mean(M3645_i3c1[:,5])-      np.mean(M3645_i3c1[:,6])-      np.mean(M3645_i3c1[:,7])-      np.mean(M3645_i3c1[:,8])-      np.mean(M3645_i3c1[:,9])-      np.mean(M3645_i3c1[:,10])-     np.mean(M3645_i3c1[:,11])-     np.mean(M3645_i3c1[:,12])   ])
M_overlap2Means.append([   np.mean(M3645_i3c08[:,5]),  np.mean(M3645_i3c09[:,5]),     np.mean(M3645_i3c095[:,5]),    np.mean(M3645_i3c098[:,5]),    np.mean(M3645_i3c1[:,5])  ])
M_overlap3Means.append([   np.mean(M3645_i3c08[:,6]),  np.mean(M3645_i3c09[:,6]),     np.mean(M3645_i3c095[:,6]),    np.mean(M3645_i3c098[:,6]),    np.mean(M3645_i3c1[:,6])  ])
M_overlap4Means.append([   np.mean(M3645_i3c08[:,7]),  np.mean(M3645_i3c09[:,7]),     np.mean(M3645_i3c095[:,7]),    np.mean(M3645_i3c098[:,7]),    np.mean(M3645_i3c1[:,7])  ])
M_overlap5Means.append([   np.mean(M3645_i3c08[:,8]),  np.mean(M3645_i3c09[:,8]),     np.mean(M3645_i3c095[:,8]),    np.mean(M3645_i3c098[:,8]),    np.mean(M3645_i3c1[:,8])  ])
M_overlap6Means.append([   np.mean(M3645_i3c08[:,9]),  np.mean(M3645_i3c09[:,9]),     np.mean(M3645_i3c095[:,9]),    np.mean(M3645_i3c098[:,9]),    np.mean(M3645_i3c1[:,9])  ])
M_overlap7Means.append([   np.mean(M3645_i3c08[:,10]), np.mean(M3645_i3c09[:,10]),    np.mean(M3645_i3c095[:,10]),   np.mean(M3645_i3c098[:,10]),   np.mean(M3645_i3c1[:,10]) ])
M_overlap8Means.append([   np.mean(M3645_i3c08[:,11]), np.mean(M3645_i3c09[:,11]),    np.mean(M3645_i3c095[:,11]),   np.mean(M3645_i3c098[:,11]),   np.mean(M3645_i3c1[:,11]) ])
M_overlap9Means.append([   np.mean(M3645_i3c08[:,12]), np.mean(M3645_i3c09[:,12]),    np.mean(M3645_i3c095[:,12]),   np.mean(M3645_i3c098[:,12]),   np.mean(M3645_i3c1[:,12]) ])
M_semOverlapStd.append([   np.std(M3645_i3c08[:,4])-   np.std(M3645_i3c08[:,5])-      np.std(M3645_i3c08[:,6])-      np.std(M3645_i3c08[:,7])-      np.std(M3645_i3c08[:,8])-     np.std(M3645_i3c08[:,9])-     np.std(M3645_i3c08[:,10])-    np.std(M3645_i3c08[:,11])-    np.std(M3645_i3c08[:,12]),
                           np.std(M3645_i3c09[:,4])-   np.std(M3645_i3c09[:,5])-      np.std(M3645_i3c09[:,6])-      np.std(M3645_i3c09[:,7])-      np.std(M3645_i3c09[:,8])-     np.std(M3645_i3c09[:,9])-     np.std(M3645_i3c09[:,10])-    np.std(M3645_i3c09[:,11])-    np.std(M3645_i3c09[:,12]),
                           np.std(M3645_i3c095[:,4])-  np.std(M3645_i3c095[:,5])-     np.std(M3645_i3c095[:,6])-     np.std(M3645_i3c095[:,7])-     np.std(M3645_i3c095[:,8])-    np.std(M3645_i3c095[:,9])-    np.std(M3645_i3c095[:,10])-   np.std(M3645_i3c095[:,11])-   np.std(M3645_i3c095[:,12]),
                           np.std(M3645_i3c098[:,4])-  np.std(M3645_i3c098[:,5])-     np.std(M3645_i3c098[:,6])-     np.std(M3645_i3c098[:,7])-     np.std(M3645_i3c098[:,8])-    np.std(M3645_i3c098[:,9])-    np.std(M3645_i3c098[:,10])-   np.std(M3645_i3c098[:,11])-   np.std(M3645_i3c098[:,12]),
                           np.std(M3645_i3c1[:,4])-    np.std(M3645_i3c1[:,5])-       np.std(M3645_i3c1[:,6])-       np.std(M3645_i3c1[:,7])-       np.std(M3645_i3c1[:,8])-      np.std(M3645_i3c1[:,9])-      np.std(M3645_i3c1[:,10])-     np.std(M3645_i3c1[:,11])-     np.std(M3645_i3c1[:,12])   ])
M_overlap2Std.append([     np.std(M3645_i3c08[:,5]),   np.std(M3645_i3c09[:,5]),      np.std(M3645_i3c095[:,5]),     np.std(M3645_i3c098[:,5]),     np.std(M3645_i3c1[:,5])  ])
M_overlap3Std.append([     np.std(M3645_i3c08[:,6]),   np.std(M3645_i3c09[:,6]),      np.std(M3645_i3c095[:,6]),     np.std(M3645_i3c098[:,6]),     np.std(M3645_i3c1[:,6])  ])
M_overlap4Std.append([     np.std(M3645_i3c08[:,7]),   np.std(M3645_i3c09[:,7]),      np.std(M3645_i3c095[:,7]),     np.std(M3645_i3c098[:,7]),     np.std(M3645_i3c1[:,7])  ])
M_overlap5Std.append([     np.std(M3645_i3c08[:,8]),   np.std(M3645_i3c09[:,8]),      np.std(M3645_i3c095[:,8]),     np.std(M3645_i3c098[:,8]),     np.std(M3645_i3c1[:,8])  ])
M_overlap6Std.append([     np.std(M3645_i3c08[:,9]),   np.std(M3645_i3c09[:,9]),      np.std(M3645_i3c095[:,9]),     np.std(M3645_i3c098[:,9]),     np.std(M3645_i3c1[:,9])  ])
M_overlap7Std.append([     np.std(M3645_i3c08[:,10]),  np.std(M3645_i3c09[:,10]),     np.std(M3645_i3c095[:,10]),    np.std(M3645_i3c098[:,10]),    np.std(M3645_i3c1[:,10]) ])
M_overlap8Std.append([     np.std(M3645_i3c08[:,11]),  np.std(M3645_i3c09[:,11]),     np.std(M3645_i3c095[:,11]),    np.std(M3645_i3c098[:,11]),    np.std(M3645_i3c1[:,11]) ])
M_overlap9Std.append([     np.std(M3645_i3c08[:,12]),  np.std(M3645_i3c09[:,12]),     np.std(M3645_i3c095[:,12]),    np.std(M3645_i3c098[:,12]),    np.std(M3645_i3c1[:,12]) ])

H_semOverlapMeans.append([ np.mean(H3645_k5[:,4])-  np.mean(H3645_k5[:,5])-     np.mean(H3645_k5[:,6])-     np.mean(H3645_k5[:,7])-    np.mean(H3645_k5[:,8])-     np.mean(H3645_k5[:,9])-     np.mean(H3645_k5[:,10])-    np.mean(H3645_k5[:,11])-    np.mean(H3645_k5[:,12]),
                           np.mean(H3645_k7[:,4])-  np.mean(H3645_k7[:,5])-     np.mean(H3645_k7[:,6])-     np.mean(H3645_k7[:,7])-    np.mean(H3645_k7[:,8])-     np.mean(H3645_k7[:,9])-     np.mean(H3645_k7[:,10])-    np.mean(H3645_k7[:,11])-    np.mean(H3645_k7[:,12]),
                           np.mean(H3645_k6[:,4])-  np.mean(H3645_k6[:,5])-     np.mean(H3645_k6[:,6])-     np.mean(H3645_k6[:,7])-    np.mean(H3645_k6[:,8])-     np.mean(H3645_k6[:,9])-     np.mean(H3645_k6[:,10])-    np.mean(H3645_k6[:,11])-    np.mean(H3645_k6[:,12]),
                           np.mean(H3645_k6[:,4])-  np.mean(H3645_k6[:,5])-     np.mean(H3645_k6[:,6])-     np.mean(H3645_k6[:,7])-    np.mean(H3645_k6[:,8])-     np.mean(H3645_k6[:,9])-     np.mean(H3645_k6[:,10])-    np.mean(H3645_k6[:,11])-    np.mean(H3645_k6[:,12]),
                           np.mean(H3645_k7[:,4])-  np.mean(H3645_k7[:,5])-     np.mean(H3645_k7[:,6])-     np.mean(H3645_k7[:,7])-    np.mean(H3645_k7[:,8])-     np.mean(H3645_k7[:,9])-     np.mean(H3645_k7[:,10])-    np.mean(H3645_k7[:,11])-    np.mean(H3645_k7[:,12])   ])
H_overlap2Means.append([   np.mean(H3645_k5[:,5]),  np.mean(H3645_k7[:,5]),     np.mean(H3645_k6[:,5]),     np.mean(H3645_k6[:,5]),    np.mean(H3645_k7[:,5])  ])
H_overlap3Means.append([   np.mean(H3645_k5[:,6]),  np.mean(H3645_k7[:,6]),     np.mean(H3645_k6[:,6]),     np.mean(H3645_k6[:,6]),    np.mean(H3645_k7[:,6])  ])
H_overlap4Means.append([   np.mean(H3645_k5[:,7]),  np.mean(H3645_k7[:,7]),     np.mean(H3645_k6[:,7]),     np.mean(H3645_k6[:,7]),    np.mean(H3645_k7[:,7])  ])
H_overlap5Means.append([   np.mean(H3645_k5[:,8]),  np.mean(H3645_k7[:,8]),     np.mean(H3645_k6[:,8]),     np.mean(H3645_k6[:,8]),    np.mean(H3645_k7[:,8])  ])
H_overlap6Means.append([   np.mean(H3645_k5[:,9]),  np.mean(H3645_k7[:,9]),     np.mean(H3645_k6[:,9]),     np.mean(H3645_k6[:,9]),    np.mean(H3645_k7[:,9])  ])
H_overlap7Means.append([   np.mean(H3645_k5[:,10]), np.mean(H3645_k7[:,10]),    np.mean(H3645_k6[:,10]),    np.mean(H3645_k6[:,10]),   np.mean(H3645_k7[:,10]) ])
H_overlap8Means.append([   np.mean(H3645_k5[:,11]), np.mean(H3645_k7[:,11]),    np.mean(H3645_k6[:,11]),    np.mean(H3645_k6[:,11]),   np.mean(H3645_k7[:,11]) ])
H_overlap9Means.append([   np.mean(H3645_k5[:,12]), np.mean(H3645_k7[:,12]),    np.mean(H3645_k6[:,12]),    np.mean(H3645_k6[:,12]),   np.mean(H3645_k7[:,12]) ])
H_semOverlapStd.append([   np.std(H3645_k5[:,4])-   np.std(H3645_k5[:,5])-      np.std(H3645_k5[:,6])-      np.std(H3645_k5[:,7])-     np.std(H3645_k5[:,8])-     np.std(H3645_k5[:,9])-     np.std(H3645_k5[:,10])-    np.std(H3645_k5[:,11])-    np.std(H3645_k5[:,12]),
                           np.std(H3645_k7[:,4])-   np.std(H3645_k7[:,5])-      np.std(H3645_k7[:,6])-      np.std(H3645_k7[:,7])-     np.std(H3645_k7[:,8])-     np.std(H3645_k7[:,9])-     np.std(H3645_k7[:,10])-    np.std(H3645_k7[:,11])-    np.std(H3645_k7[:,12]),
                           np.std(H3645_k6[:,4])-   np.std(H3645_k6[:,5])-      np.std(H3645_k6[:,6])-      np.std(H3645_k6[:,7])-     np.std(H3645_k6[:,8])-     np.std(H3645_k6[:,9])-     np.std(H3645_k6[:,10])-    np.std(H3645_k6[:,11])-    np.std(H3645_k6[:,12]),
                           np.std(H3645_k6[:,4])-   np.std(H3645_k6[:,5])-      np.std(H3645_k6[:,6])-      np.std(H3645_k6[:,7])-     np.std(H3645_k6[:,8])-     np.std(H3645_k6[:,9])-     np.std(H3645_k6[:,10])-    np.std(H3645_k6[:,11])-    np.std(H3645_k6[:,12]),
                           np.std(H3645_k7[:,4])-   np.std(H3645_k7[:,5])-      np.std(H3645_k7[:,6])-      np.std(H3645_k7[:,7])-     np.std(H3645_k7[:,8])-     np.std(H3645_k7[:,9])-     np.std(H3645_k7[:,10])-    np.std(H3645_k7[:,11])-    np.std(H3645_k7[:,12])   ])
H_overlap2Std.append([     np.std(H3645_k5[:,5]),   np.std(H3645_k7[:,5]),      np.std(H3645_k6[:,5]),      np.std(H3645_k6[:,5]),     np.std(H3645_k7[:,5])  ])
H_overlap3Std.append([     np.std(H3645_k5[:,6]),   np.std(H3645_k7[:,6]),      np.std(H3645_k6[:,6]),      np.std(H3645_k6[:,6]),     np.std(H3645_k7[:,6])  ])
H_overlap4Std.append([     np.std(H3645_k5[:,7]),   np.std(H3645_k7[:,7]),      np.std(H3645_k6[:,7]),      np.std(H3645_k6[:,7]),     np.std(H3645_k7[:,7])  ])
H_overlap5Std.append([     np.std(H3645_k5[:,8]),   np.std(H3645_k7[:,8]),      np.std(H3645_k6[:,8]),      np.std(H3645_k6[:,8]),     np.std(H3645_k7[:,8])  ])
H_overlap6Std.append([     np.std(H3645_k5[:,9]),   np.std(H3645_k7[:,9]),      np.std(H3645_k6[:,9]),      np.std(H3645_k6[:,9]),     np.std(H3645_k7[:,9])  ])
H_overlap7Std.append([     np.std(H3645_k5[:,10]),  np.std(H3645_k7[:,10]),     np.std(H3645_k6[:,10]),     np.std(H3645_k6[:,10]),    np.std(H3645_k7[:,10]) ])
H_overlap8Std.append([     np.std(H3645_k5[:,11]),  np.std(H3645_k7[:,11]),     np.std(H3645_k6[:,11]),     np.std(H3645_k6[:,11]),    np.std(H3645_k7[:,11]) ])
H_overlap9Std.append([     np.std(H3645_k5[:,12]),  np.std(H3645_k7[:,12]),     np.std(H3645_k6[:,12]),     np.std(H3645_k6[:,12]),    np.std(H3645_k7[:,12]) ])

E_semOverlapMeans = np.array(E_semOverlapMeans)
E_overlap2Means = np.array(E_overlap2Means)
E_overlap3Means = np.array(E_overlap3Means)
E_overlap4Means = np.array(E_overlap4Means)
E_overlap5Means = np.array(E_overlap5Means)
E_overlap6Means = np.array(E_overlap6Means)
E_overlap7Means = np.array(E_overlap7Means)
E_overlap8Means = np.array(E_overlap8Means)
E_overlap9Means = np.array(E_overlap9Means)

M_semOverlapMeans = np.array(M_semOverlapMeans)
M_overlap2Means = np.array(M_overlap2Means)
M_overlap3Means = np.array(M_overlap3Means)
M_overlap4Means = np.array(M_overlap4Means)
M_overlap5Means = np.array(M_overlap5Means)
M_overlap6Means = np.array(M_overlap6Means)
M_overlap7Means = np.array(M_overlap7Means)
M_overlap8Means = np.array(M_overlap8Means)
M_overlap9Means = np.array(M_overlap9Means)
M_semOverlapStd = np.array(M_semOverlapStd)
M_overlap2Std = np.array(M_overlap2Std)
M_overlap3Std = np.array(M_overlap3Std)
M_overlap4Std = np.array(M_overlap4Std)
M_overlap5Std = np.array(M_overlap5Std)
M_overlap6Std = np.array(M_overlap6Std)
M_overlap7Std = np.array(M_overlap7Std)
M_overlap8Std = np.array(M_overlap8Std)
M_overlap9Std = np.array(M_overlap9Std)

H_semOverlapMeans = np.array(H_semOverlapMeans)
H_overlap2Means = np.array(H_overlap2Means)
H_overlap3Means = np.array(H_overlap3Means)
H_overlap4Means = np.array(H_overlap4Means)
H_overlap5Means = np.array(H_overlap5Means)
H_overlap6Means = np.array(H_overlap6Means)
H_overlap7Means = np.array(H_overlap7Means)
H_overlap8Means = np.array(H_overlap8Means)
H_overlap9Means = np.array(H_overlap9Means)
H_semOverlapStd = np.array(H_semOverlapStd)
H_overlap2Std = np.array(H_overlap2Std)
H_overlap3Std = np.array(H_overlap3Std)
H_overlap4Std = np.array(H_overlap4Std)
H_overlap5Std = np.array(H_overlap5Std)
H_overlap6Std = np.array(H_overlap6Std)
H_overlap7Std = np.array(H_overlap7Std)
H_overlap8Std = np.array(H_overlap8Std)
H_overlap9Std = np.array(H_overlap9Std)

fig, axs = plt.subplots(1, 4, sharey=True)
labels = [80, 90, 95, 98, 100]
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars
colors = ['#15B01A', '#FFFF14', '#FFA500','#F97306','#FE420F', '#FF6347', '#EF4026', '#FF0000', '#E50000']

###   Cenário Brum1215
axs[0].bar(x - width,  E_semOverlapMeans[0],    width,                              label='E-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0].bar(x - width,  E_overlap2Means[0],      width,                                                      color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0])
axs[0].bar(x - width,  E_overlap3Means[0],      width,                                                      color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0])
axs[0].bar(x - width,  E_overlap4Means[0],      width,                                                      color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0])
axs[0].bar(x - width,  E_overlap5Means[0],      width,                                                      color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0]+E_overlap4Means[0])
axs[0].bar(x - width,  E_overlap6Means[0],      width,                                                      color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0]+E_overlap4Means[0]+E_overlap5Means[0])
axs[0].bar(x - width,  E_overlap7Means[0],      width,                                                      color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0]+E_overlap4Means[0]+E_overlap5Means[0]+E_overlap6Means[0])
axs[0].bar(x - width,  E_overlap8Means[0],      width,                                                      color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0]+E_overlap4Means[0]+E_overlap5Means[0]+E_overlap6Means[0]+E_overlap7Means[0])
axs[0].bar(x - width,  E_overlap9Means[0],      width,                                                      color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[0]+E_overlap2Means[0]+E_overlap3Means[0]+E_overlap4Means[0]+E_overlap5Means[0]+E_overlap6Means[0]+E_overlap7Means[0]+E_overlap8Means[0])

axs[0].bar(x ,         M_semOverlapMeans[0],    width, yerr=M_semOverlapStd[0] ,    label='M-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0].bar(x ,         M_overlap2Means[0],      width, yerr=M_overlap2Std[0] ,                              color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0])
axs[0].bar(x ,         M_overlap3Means[0],      width, yerr=M_overlap3Std[0] ,                              color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0])
axs[0].bar(x ,         M_overlap4Means[0],      width, yerr=M_overlap4Std[0] ,                              color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0])
axs[0].bar(x ,         M_overlap5Means[0],      width, yerr=M_overlap5Std[0] ,                              color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0]+M_overlap4Means[0])
axs[0].bar(x ,         M_overlap6Means[0],      width, yerr=M_overlap6Std[0] ,                              color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0]+M_overlap4Means[0]+M_overlap5Means[0])
axs[0].bar(x ,         M_overlap7Means[0],      width, yerr=M_overlap7Std[0] ,                              color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0]+M_overlap4Means[0]+M_overlap5Means[0]+M_overlap6Means[0])
axs[0].bar(x ,         M_overlap8Means[0],      width, yerr=M_overlap8Std[0] ,                              color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0]+M_overlap4Means[0]+M_overlap5Means[0]+M_overlap6Means[0]+M_overlap7Means[0])
axs[0].bar(x ,         M_overlap9Means[0],      width, yerr=M_overlap9Std[0] ,                              color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[0]+M_overlap2Means[0]+M_overlap3Means[0]+M_overlap4Means[0]+M_overlap5Means[0]+M_overlap6Means[0]+M_overlap7Means[0]+M_overlap8Means[0])

axs[0].bar(x + width,  H_semOverlapMeans[0],   width, yerr=H_semOverlapStd[0] ,     label='Heuristic',      color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)
axs[0].bar(x + width,  H_overlap2Means[0],     width, yerr=H_overlap2Std[0] ,                               color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0])
axs[0].bar(x + width,  H_overlap3Means[0],     width, yerr=H_overlap3Std[0] ,                               color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0])
axs[0].bar(x + width,  H_overlap4Means[0],     width, yerr=H_overlap4Std[0] ,                               color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0])
axs[0].bar(x + width,  H_overlap5Means[0],     width, yerr=H_overlap5Std[0] ,                               color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0]+H_overlap4Means[0])
axs[0].bar(x + width,  H_overlap6Means[0],     width, yerr=H_overlap6Std[0] ,                               color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0]+H_overlap4Means[0]+H_overlap5Means[0])
axs[0].bar(x + width,  H_overlap7Means[0],     width, yerr=H_overlap7Std[0] ,                               color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0]+H_overlap4Means[0]+H_overlap5Means[0]+H_overlap6Means[0])
axs[0].bar(x + width,  H_overlap8Means[0],     width, yerr=H_overlap8Std[0] ,                               color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0]+H_overlap4Means[0]+H_overlap5Means[0]+H_overlap6Means[0]+H_overlap7Means[0])
axs[0].bar(x + width,  H_overlap9Means[0],     width, yerr=H_overlap9Std[0] ,                               color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[0]+H_overlap2Means[0]+H_overlap3Means[0]+H_overlap4Means[0]+H_overlap5Means[0]+H_overlap6Means[0]+H_overlap7Means[0]+H_overlap8Means[0])


###   Cenário Brum2025   ###
axs[1].bar(x - width,  E_semOverlapMeans[1],    width,                              label='E-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1].bar(x - width,  E_overlap2Means[1],      width,                                                      color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1])
axs[1].bar(x - width,  E_overlap3Means[1],      width,                                                      color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1])
axs[1].bar(x - width,  E_overlap4Means[1],      width,                                                      color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1])
axs[1].bar(x - width,  E_overlap5Means[1],      width,                                                      color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1]+E_overlap4Means[1])
axs[1].bar(x - width,  E_overlap6Means[1],      width,                                                      color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1]+E_overlap4Means[1]+E_overlap5Means[1])
axs[1].bar(x - width,  E_overlap7Means[1],      width,                                                      color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1]+E_overlap4Means[1]+E_overlap5Means[1]+E_overlap6Means[1])
axs[1].bar(x - width,  E_overlap8Means[1],      width,                                                      color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1]+E_overlap4Means[1]+E_overlap5Means[1]+E_overlap6Means[1]+E_overlap7Means[1])
axs[1].bar(x - width,  E_overlap9Means[1],      width,                                                      color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[1]+E_overlap2Means[1]+E_overlap3Means[1]+E_overlap4Means[1]+E_overlap5Means[1]+E_overlap6Means[1]+E_overlap7Means[1]+E_overlap8Means[1])

axs[1].bar(x ,         M_semOverlapMeans[1],    width, yerr=M_semOverlapStd[1] ,    label='M-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1].bar(x ,         M_overlap2Means[1],      width, yerr=M_overlap2Std[1] ,                              color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1])
axs[1].bar(x ,         M_overlap3Means[1],      width, yerr=M_overlap3Std[1] ,                              color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1])
axs[1].bar(x ,         M_overlap4Means[1],      width, yerr=M_overlap4Std[1] ,                              color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1])
axs[1].bar(x ,         M_overlap5Means[1],      width, yerr=M_overlap5Std[1] ,                              color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1]+M_overlap4Means[1])
axs[1].bar(x ,         M_overlap6Means[1],      width, yerr=M_overlap6Std[1] ,                              color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1]+M_overlap4Means[1]+M_overlap5Means[1])
axs[1].bar(x ,         M_overlap7Means[1],      width, yerr=M_overlap7Std[1] ,                              color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1]+M_overlap4Means[1]+M_overlap5Means[1]+M_overlap6Means[1])
axs[1].bar(x ,         M_overlap8Means[1],      width, yerr=M_overlap8Std[1] ,                              color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1]+M_overlap4Means[1]+M_overlap5Means[1]+M_overlap6Means[1]+M_overlap7Means[1])
axs[1].bar(x ,         M_overlap9Means[1],      width, yerr=M_overlap9Std[1] ,                              color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[1]+M_overlap2Means[1]+M_overlap3Means[1]+M_overlap4Means[1]+M_overlap5Means[1]+M_overlap6Means[1]+M_overlap7Means[1]+M_overlap8Means[1])

axs[1].bar(x + width,  H_semOverlapMeans[1],   width, yerr=H_semOverlapStd[1] ,     label='Heuristic',      color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)
axs[1].bar(x + width,  H_overlap2Means[1],     width, yerr=H_overlap2Std[1] ,                               color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1])
axs[1].bar(x + width,  H_overlap3Means[1],     width, yerr=H_overlap3Std[1] ,                               color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1])
axs[1].bar(x + width,  H_overlap4Means[1],     width, yerr=H_overlap4Std[1] ,                               color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1])
axs[1].bar(x + width,  H_overlap5Means[1],     width, yerr=H_overlap5Std[1] ,                               color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1]+H_overlap4Means[1])
axs[1].bar(x + width,  H_overlap6Means[1],     width, yerr=H_overlap6Std[1] ,                               color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1]+H_overlap4Means[1]+H_overlap5Means[1])
axs[1].bar(x + width,  H_overlap7Means[1],     width, yerr=H_overlap7Std[1] ,                               color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1]+H_overlap4Means[1]+H_overlap5Means[1]+H_overlap6Means[1])
axs[1].bar(x + width,  H_overlap8Means[1],     width, yerr=H_overlap8Std[1] ,                               color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1]+H_overlap4Means[1]+H_overlap5Means[1]+H_overlap6Means[1]+H_overlap7Means[1])
axs[1].bar(x + width,  H_overlap9Means[1],     width, yerr=H_overlap9Std[1] ,                               color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[1]+H_overlap2Means[1]+H_overlap3Means[1]+H_overlap4Means[1]+H_overlap5Means[1]+H_overlap6Means[1]+H_overlap7Means[1]+H_overlap8Means[1])


###   Cenário Brum2430   ###
axs[2].bar(x - width,  E_semOverlapMeans[2],    width,                              label='E-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[2].bar(x - width,  E_overlap2Means[2],      width,                                                      color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2])
axs[2].bar(x - width,  E_overlap3Means[2],      width,                                                      color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2])
axs[2].bar(x - width,  E_overlap4Means[2],      width,                                                      color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2])
axs[2].bar(x - width,  E_overlap5Means[2],      width,                                                      color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2]+E_overlap4Means[2])
axs[2].bar(x - width,  E_overlap6Means[2],      width,                                                      color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2]+E_overlap4Means[2]+E_overlap5Means[2])
axs[2].bar(x - width,  E_overlap7Means[2],      width,                                                      color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2]+E_overlap4Means[2]+E_overlap5Means[2]+E_overlap6Means[2])
axs[2].bar(x - width,  E_overlap8Means[2],      width,                                                      color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2]+E_overlap4Means[2]+E_overlap5Means[2]+E_overlap6Means[2]+E_overlap7Means[2])
axs[2].bar(x - width,  E_overlap9Means[2],      width,                                                      color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[2]+E_overlap2Means[2]+E_overlap3Means[2]+E_overlap4Means[2]+E_overlap5Means[2]+E_overlap6Means[2]+E_overlap7Means[2]+E_overlap8Means[2])

axs[2].bar(x ,         M_semOverlapMeans[2],    width, yerr=M_semOverlapStd[2] ,    label='M-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[2].bar(x ,         M_overlap2Means[2],      width, yerr=M_overlap2Std[2] ,                              color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2])
axs[2].bar(x ,         M_overlap3Means[2],      width, yerr=M_overlap3Std[2] ,                              color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2])
axs[2].bar(x ,         M_overlap4Means[2],      width, yerr=M_overlap4Std[2] ,                              color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2])
axs[2].bar(x ,         M_overlap5Means[2],      width, yerr=M_overlap5Std[2] ,                              color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2]+M_overlap4Means[2])
axs[2].bar(x ,         M_overlap6Means[2],      width, yerr=M_overlap6Std[2] ,                              color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2]+M_overlap4Means[2]+M_overlap5Means[2])
axs[2].bar(x ,         M_overlap7Means[2],      width, yerr=M_overlap7Std[2] ,                              color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2]+M_overlap4Means[2]+M_overlap5Means[2]+M_overlap6Means[2])
axs[2].bar(x ,         M_overlap8Means[2],      width, yerr=M_overlap8Std[2] ,                              color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2]+M_overlap4Means[2]+M_overlap5Means[2]+M_overlap6Means[2]+M_overlap7Means[2])
axs[2].bar(x ,         M_overlap9Means[2],      width, yerr=M_overlap9Std[2] ,                              color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[2]+M_overlap2Means[2]+M_overlap3Means[2]+M_overlap4Means[2]+M_overlap5Means[2]+M_overlap6Means[2]+M_overlap7Means[2]+M_overlap8Means[2])

axs[2].bar(x + width,  H_semOverlapMeans[2],   width, yerr=H_semOverlapStd[2] ,     label='Heuristic',      color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)
axs[2].bar(x + width,  H_overlap2Means[2],     width, yerr=H_overlap2Std[2] ,                               color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2])
axs[2].bar(x + width,  H_overlap3Means[2],     width, yerr=H_overlap3Std[2] ,                               color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2])
axs[2].bar(x + width,  H_overlap4Means[2],     width, yerr=H_overlap4Std[2] ,                               color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2])
axs[2].bar(x + width,  H_overlap5Means[2],     width, yerr=H_overlap5Std[2] ,                               color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2]+H_overlap4Means[2])
axs[2].bar(x + width,  H_overlap6Means[2],     width, yerr=H_overlap6Std[2] ,                               color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2]+H_overlap4Means[2]+H_overlap5Means[2])
axs[2].bar(x + width,  H_overlap7Means[2],     width, yerr=H_overlap7Std[2] ,                               color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2]+H_overlap4Means[2]+H_overlap5Means[2]+H_overlap6Means[2])
axs[2].bar(x + width,  H_overlap8Means[2],     width, yerr=H_overlap8Std[2] ,                               color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2]+H_overlap4Means[2]+H_overlap5Means[2]+H_overlap6Means[2]+H_overlap7Means[2])
axs[2].bar(x + width,  H_overlap9Means[2],     width, yerr=H_overlap9Std[2] ,                               color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[2]+H_overlap2Means[2]+H_overlap3Means[2]+H_overlap4Means[2]+H_overlap5Means[2]+H_overlap6Means[2]+H_overlap7Means[2]+H_overlap8Means[2])


###   Cenário Brum3645   ###
axs[3].bar(x - width,  E_semOverlapMeans[3],    width,                              label='E-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[3].bar(x - width,  E_overlap2Means[3],      width,                                                      color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3])
axs[3].bar(x - width,  E_overlap3Means[3],      width,                                                      color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3])
axs[3].bar(x - width,  E_overlap4Means[3],      width,                                                      color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3])
axs[3].bar(x - width,  E_overlap5Means[3],      width,                                                      color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3]+E_overlap4Means[3])
axs[3].bar(x - width,  E_overlap6Means[3],      width,                                                      color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3]+E_overlap4Means[3]+E_overlap5Means[3])
axs[3].bar(x - width,  E_overlap7Means[3],      width,                                                      color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3]+E_overlap4Means[3]+E_overlap5Means[3]+E_overlap6Means[3])
axs[3].bar(x - width,  E_overlap8Means[3],      width,                                                      color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3]+E_overlap4Means[3]+E_overlap5Means[3]+E_overlap6Means[3]+E_overlap7Means[3])
axs[3].bar(x - width,  E_overlap9Means[3],      width,                                                      color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7, bottom=E_semOverlapMeans[3]+E_overlap2Means[3]+E_overlap3Means[3]+E_overlap4Means[3]+E_overlap5Means[3]+E_overlap6Means[3]+E_overlap7Means[3]+E_overlap8Means[3])

axs[3].bar(x ,         M_semOverlapMeans[3],    width, yerr=M_semOverlapStd[3] ,    label='M-ALLOCATOR',    color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[3].bar(x ,         M_overlap2Means[3],      width, yerr=M_overlap2Std[3] ,                              color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3])
axs[3].bar(x ,         M_overlap3Means[3],      width, yerr=M_overlap3Std[3] ,                              color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3])
axs[3].bar(x ,         M_overlap4Means[3],      width, yerr=M_overlap4Std[3] ,                              color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3])
axs[3].bar(x ,         M_overlap5Means[3],      width, yerr=M_overlap5Std[3] ,                              color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3]+M_overlap4Means[3])
axs[3].bar(x ,         M_overlap6Means[3],      width, yerr=M_overlap6Std[3] ,                              color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3]+M_overlap4Means[3]+M_overlap5Means[3])
axs[3].bar(x ,         M_overlap7Means[3],      width, yerr=M_overlap7Std[3] ,                              color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3]+M_overlap4Means[3]+M_overlap5Means[3]+M_overlap6Means[3])
axs[3].bar(x ,         M_overlap8Means[3],      width, yerr=M_overlap8Std[3] ,                              color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3]+M_overlap4Means[3]+M_overlap5Means[3]+M_overlap6Means[3]+M_overlap7Means[3])
axs[3].bar(x ,         M_overlap9Means[3],      width, yerr=M_overlap9Std[3] ,                              color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7, bottom=M_semOverlapMeans[3]+M_overlap2Means[3]+M_overlap3Means[3]+M_overlap4Means[3]+M_overlap5Means[3]+M_overlap6Means[3]+M_overlap7Means[3]+M_overlap8Means[3])

axs[3].bar(x + width,  H_semOverlapMeans[3],   width, yerr=H_semOverlapStd[3] ,     label='Heuristic',      color=colors[0], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)
axs[3].bar(x + width,  H_overlap2Means[3],     width, yerr=H_overlap2Std[3] ,                               color=colors[1], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3])
axs[3].bar(x + width,  H_overlap3Means[3],     width, yerr=H_overlap3Std[3] ,                               color=colors[2], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3])
axs[3].bar(x + width,  H_overlap4Means[3],     width, yerr=H_overlap4Std[3] ,                               color=colors[3], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3])
axs[3].bar(x + width,  H_overlap5Means[3],     width, yerr=H_overlap5Std[3] ,                               color=colors[4], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3]+H_overlap4Means[3])
axs[3].bar(x + width,  H_overlap6Means[3],     width, yerr=H_overlap6Std[3] ,                               color=colors[5], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3]+H_overlap4Means[3]+H_overlap5Means[3])
axs[3].bar(x + width,  H_overlap7Means[3],     width, yerr=H_overlap7Std[3] ,                               color=colors[6], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3]+H_overlap4Means[3]+H_overlap5Means[3]+H_overlap6Means[3])
axs[3].bar(x + width,  H_overlap8Means[3],     width, yerr=H_overlap8Std[3] ,                               color=colors[7], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3]+H_overlap4Means[3]+H_overlap5Means[3]+H_overlap6Means[3]+H_overlap7Means[3])
axs[3].bar(x + width,  H_overlap9Means[3],     width, yerr=H_overlap9Std[3] ,                               color=colors[8], edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7, bottom=H_semOverlapMeans[3]+H_overlap2Means[3]+H_overlap3Means[3]+H_overlap4Means[3]+H_overlap5Means[3]+H_overlap6Means[3]+H_overlap7Means[3]+H_overlap8Means[3])

axs[0].grid(axis='y', alpha=0.4)
axs[1].grid(axis='y', alpha=0.4)
axs[2].grid(axis='y', alpha=0.4)
axs[3].grid(axis='y', alpha=0.4)

axs[0].set_title(cenarios[0])
axs[1].set_title(cenarios[1])
axs[2].set_title(cenarios[2])
axs[3].set_title(cenarios[3])

axs[0].set_ylabel('Overlap Area (km\u00b2)')

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Coverage Consensus (%)")

axs[0].set_xticks(x)
axs[1].set_xticks(x)
axs[2].set_xticks(x)
axs[3].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[1].set_xticklabels(labels)
axs[2].set_xticklabels(labels)
axs[3].set_xticklabels(labels)

axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
#plt.colorbar()
plt.subplots_adjust(left=0.04, bottom=0.15)
plt.show()