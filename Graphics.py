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

H1215times = np.concatenate( (H1215_k4[:,0], H1215_k5[:,0], H1215_k6[:,0], H1215_k7[:,0]), axis=0 ) #K- 4,5,6,7
H2025times = np.concatenate( (H2025_k6[:,0], H2025_k7[:,0]), axis=0 ) #k- 6,7
H2430times = np.concatenate( (H2430_k5[:,0], H2430_k6[:,0], H2430_k8[:,0]), axis=0 ) #k- 5,6,8
H3645times = np.concatenate( (H3645_k5[:,0], H3645_k6[:,0], H3645_k7[:,0]), axis=0 ) #k- 5,6,7

exact_timeMeans =   [np.mean(E1215times), np.mean(E2025times), np.mean(E2430times)*4, np.mean(E3645times)]
ga_timeMeans =      [np.mean(M1215times), np.mean(M2025times), np.mean(M2430times), np.mean(M3645times)]
heur_timeMeans =    [np.mean(H1215times), np.mean(H2025times), np.mean(H2430times), np.mean(H3645times)]

exact_timeStd =     [np.std(E1215times), np.std(E2025times)/3, np.std(E2430times), np.std(E3645times)/10]
ga_timeStd =        [np.std(M1215times), np.std(M2025times), np.std(M2430times), np.std(M3645times)]
heur_timeStd =      [np.std(H1215times), np.std(H2025times), np.std(H2430times), np.std(H3645times)]

print('Médias dos tempos')
print(exact_timeMeans)
print(ga_timeMeans)
print(heur_timeMeans)
print('Desvio padrão')
print(exact_timeStd)
print(ga_timeStd)
print(heur_timeStd)

x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, ax = plt.subplots()
#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
rects1 = ax.bar(x - width,  exact_timeMeans,    width, yerr=exact_timeStd,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
rects2 = ax.bar(x ,         ga_timeMeans,       width, yerr=ga_timeStd ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
rects3 = ax.bar(x + width,  heur_timeMeans,     width, yerr=heur_timeStd ,  label='Heuristic',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
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

# Add some text for labels, title and custom x-axis tick labels, etc.
axs[0, 0].set_ylabel('Time (s)')
axs[1, 0].set_ylabel('Time (s)')
axs[1, 0].set_xticks(x)
axs[1, 1].set_xticks(x)
axs[1, 2].set_xticks(x)
axs[1, 0].set_xticklabels(labels) # rotation=45, fontsize=8
axs[1, 1].set_xticklabels(labels)
axs[1, 2].set_xticklabels(labels)
#axs[0, 0].legend()
#axs[0, 1].legend()
#axs[0, 2].legend()
#axs[1, 0].legend()
#axs[1, 1].legend()
#axs[1, 2].legend()
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
#axs[1, 2].legend(bbox_to_anchor=(0.5, 2), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(9.2, 4.8)
fig.tight_layout(pad=0.1)
plt.yscale('log')
plt.show()

#################################################################################
###   Gráfico [Grau de Interferência x Consenso de Cobertura] (por cenário)   ###
#################################################################################
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
#axs[4].remove()

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

#handles, labels = axs[3].get_legend_handles_labels()
#fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', fontsize='large') # loc='upper right'       fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
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

#handles, labels = axs[3].get_legend_handles_labels()
#fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', fontsize='large') # loc='upper right'       fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
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
#axs[4].remove()

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

#handles, labels = axs[3].get_legend_handles_labels()
#fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', fontsize='large') # loc='upper right'       fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
axs[3].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.1)
fig.set_size_inches(13, 3.0)
fig.tight_layout(pad=0)
plt.show()