import matplotlib.pyplot as plt
import numpy as np

###########################################################################################
###   Gráficos de barras médias/desvio padrão dos tempos de processamento consolidado   ###
###########################################################################################
labels = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645', 'Brum4050']
exact_timeMeans =   [20, 34, 30, 35, 27]
ga_timeMeans =      [25, 32, 34, 20, 25]
heur_timeMeans =    [18, 20, 22, 24, 26]

exact_timeStd =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd =      [0.6, 0.5, 0.8, 1.3, 0.8]

x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, ax = plt.subplots()
#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
rects1 = ax.bar(x - width,  exact_timeMeans,    width, yerr=exact_timeStd,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
rects2 = ax.bar(x ,         ga_timeMeans,       width, yerr=ga_timeStd ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
rects3 = ax.bar(x + width,  heur_timeMeans,     width, yerr=heur_timeStd ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

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
plt.show()

####################################################################################################
###   Gráficos de barras médias/desvio padrão dos tempos de processamento para cada Coverage %   ###
####################################################################################################
exact_timeMeans_c08 =   [20, 34, 30, 35, 27]
ga_timeMeans_c08 =      [25, 32, 34, 20, 25]
heur_timeMeans_c08 =    [18, 20, 22, 24, 26]

exact_timeStd_c08 =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd_c08 =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd_c08 =      [0.6, 0.5, 0.8, 1.3, 0.8]

exact_timeMeans_c09 =   [20, 34, 30, 35, 27]
ga_timeMeans_c09 =      [25, 32, 34, 20, 25]
heur_timeMeans_c09 =    [18, 20, 22, 24, 26]

exact_timeStd_c09 =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd_c09 =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd_c09 =      [0.6, 0.5, 0.8, 1.3, 0.8]

exact_timeMeans_c095 =   [20, 34, 30, 35, 27]
ga_timeMeans_c095 =      [25, 32, 34, 20, 25]
heur_timeMeans_c095 =    [18, 20, 22, 24, 26]

exact_timeStd_c095 =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd_c095 =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd_c095 =      [0.6, 0.5, 0.8, 1.3, 0.8]

exact_timeMeans_c098 =   [20, 34, 30, 35, 27]
ga_timeMeans_c098 =      [25, 32, 34, 20, 25]
heur_timeMeans_c098 =    [18, 20, 22, 24, 26]

exact_timeStd_c098 =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd_c098 =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd_c098 =      [0.6, 0.5, 0.8, 1.3, 0.8]

exact_timeMeans_c1 =   [20, 34, 30, 35, 27]
ga_timeMeans_c1 =      [25, 32, 34, 20, 25]
heur_timeMeans_c1 =    [18, 20, 22, 24, 26]

exact_timeStd_c1 =     [1, 2, 0.8, 2.3, 1.2]
ga_timeStd_c1 =        [1.5, 3, 1.8, 2.5, 3.2]
heur_timeStd_c1 =      [0.6, 0.5, 0.8, 1.3, 0.8]

labels = ['Brum\n1215', 'Brum\n2025', 'Brum\n2430', 'Brum\n3645', 'Brum\n4050']
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
#      Axes.bar(x,          height,         width=0.8, bottom=None, *, align='center', data=None, **kwargs)
axs[0, 0].bar(x - width,  exact_timeMeans_c08,    width, yerr=exact_timeStd_c08,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0, 0].bar(x ,         ga_timeMeans_c08,       width, yerr=ga_timeStd_c08 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0, 0].bar(x + width,  heur_timeMeans_c08,     width, yerr=heur_timeStd_c08 ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[0, 1].bar(x - width,  exact_timeMeans_c09,    width, yerr=exact_timeStd_c09,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[0, 1].bar(x ,         ga_timeMeans_c09,       width, yerr=ga_timeStd_c09 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[0, 1].bar(x + width,  heur_timeMeans_c09,     width, yerr=heur_timeStd_c09 ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 0].bar(x - width,  exact_timeMeans_c095,    width, yerr=exact_timeStd_c095,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 0].bar(x ,         ga_timeMeans_c095,       width, yerr=ga_timeStd_c095 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 0].bar(x + width,  heur_timeMeans_c095,     width, yerr=heur_timeStd_c095 ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 1].bar(x - width,  exact_timeMeans_c098,    width, yerr=exact_timeStd_c098,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 1].bar(x ,         ga_timeMeans_c098,       width, yerr=ga_timeStd_c098 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 1].bar(x + width,  heur_timeMeans_c098,     width, yerr=heur_timeStd_c098 ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

axs[1, 2].bar(x - width,  exact_timeMeans_c1,    width, yerr=exact_timeStd_c1,  label='E-ALLOCATOR',    color='r', edgecolor='black', linewidth=1.0, capsize=3, hatch='////', alpha=0.7)
axs[1, 2].bar(x ,         ga_timeMeans_c1,       width, yerr=ga_timeStd_c1 ,    label='M-ALLOCATOR',    color='b', edgecolor='black', linewidth=1.0, capsize=3, hatch='....', alpha=0.7)
axs[1, 2].bar(x + width,  heur_timeMeans_c1,     width, yerr=heur_timeStd_c1 ,  label='H-ALLOCATOR',    color='y', edgecolor='black', linewidth=1.0, capsize=3, hatch='xxxx', alpha=0.7)

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
fig.legend(handles, labels, loc='upper right', fontsize='large') #fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
fig.set_size_inches(9.2, 4.8)
fig.tight_layout(pad=0.1)
plt.show()

#######################################################################
###   Gráfico [Valor Ótimo x Consenso de Cobertura] (por cenário)   ###
#######################################################################
x = [0.8, 0.9, 0.95, 0.98, 1]
cenarios = ['Brum1215', 'Brum2025', 'Brum2430', 'Brum3645', 'Brum4050']
exact_fitValues = []
ga_fitMeans = []
heur_fitMeans = []
ga_fitStd = []
heur_fitStd =[]

for c in range(len(cenarios)):
    exact_fitValues.append([-0.08, -0.1, -0.111, -0.121, -0.132])
    ga_fitMeans.append([-0.116, -0.15, -0.17, -0.191, -0.195])
    heur_fitMeans.append([-0.146, -0.17, -0.2, -0.21, -0.23])

    ga_fitStd.append([-0.02, -0.016, -0.011, -0.021, -0.032])
    heur_fitStd.append([-0.005, -0.008, -0.01, -0.005, -0.006])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].plot(x, exact_fitValues[0], marker='s', color='r', label="E-ALLOCATOR")
axs[0, 0].errorbar(x, ga_fitMeans[0],     ga_fitStd[0],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[0, 0].errorbar(x, heur_fitMeans[0],   heur_fitStd[0], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="H-ALLOCATOR")

axs[0, 1].plot(x, exact_fitValues[1], marker='s', color='r', label="E-ALLOCATOR")
axs[0, 1].errorbar(x, ga_fitMeans[1],     ga_fitStd[1],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[0, 1].errorbar(x, heur_fitMeans[1],   heur_fitStd[1], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="H-ALLOCATOR")

axs[1, 0].plot(x, exact_fitValues[2], marker='s', color='r', label="E-ALLOCATOR")
axs[1, 0].errorbar(x, ga_fitMeans[2],     ga_fitStd[2],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[1, 0].errorbar(x, heur_fitMeans[2],   heur_fitStd[2], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="H-ALLOCATOR")

axs[1, 1].plot(x, exact_fitValues[3], marker='s', color='r', label="E-ALLOCATOR")
axs[1, 1].errorbar(x, ga_fitMeans[3],     ga_fitStd[3],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[1, 1].errorbar(x, heur_fitMeans[3],   heur_fitStd[3], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="H-ALLOCATOR")

axs[1, 2].plot(x, exact_fitValues[4], marker='s', color='r', label="E-ALLOCATOR")
axs[1, 2].errorbar(x, ga_fitMeans[4],     ga_fitStd[4],   color='b',  fmt='o', ls=':', capsize=8, markersize=7, markerfacecolor='w', capthick=2, label="M-ALLOCATOR")
axs[1, 2].errorbar(x, heur_fitMeans[4],   heur_fitStd[4], color='y',  fmt='P', ls='-.', capsize=5, markersize=8, markerfacecolor='w', label="H-ALLOCATOR")

axs[0, 2].remove()
axs[0, 0].grid(alpha=0.4)
axs[0, 1].grid(alpha=0.4)
#axs[0, 2].grid(axis='y', alpha=0.4)
axs[1, 0].grid(alpha=0.4)
axs[1, 1].grid(alpha=0.4)
axs[1, 2].grid(alpha=0.4)
axs[0, 0].set_title(cenarios[0])
axs[0, 1].set_title(cenarios[1])
axs[1, 0].set_title(cenarios[2])
axs[1, 1].set_title(cenarios[3])
axs[1, 2].set_title(cenarios[4])

#axs[0, 0].set_ylabel('Interference')
#axs[1, 0].set_ylabel('Interference')
#axs[1, 0].set_xlabel('Coverage')
#xs[1, 1].set_xlabel('Coverage')
#xs[1, 2].set_xlabel('Coverage')


handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize='large') #fontsize={'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
fig.set_size_inches(9.2, 4.8)
fig.tight_layout(pad=0.1)
plt.show()