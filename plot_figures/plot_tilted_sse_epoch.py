import json
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})

num_subsample = 10
k = 4
t = 0.5
lr = 0.03
epoch_list = [1, 3, 5, 7, 9]
dataset_names = ['athlete','bank','census','diabetes','recruitment','spanish','student','3d']
ifkm_tilted_sse_t = {}
for dataset_name in dataset_names:
    ifkm_tilted_sse_epoch = []
    for num_epoch in epoch_list:
        ifkm_tilted_sse_id = []
        for subsample_id in range(num_subsample):
            file_path = ('../ifkm_codes/output/'+dataset_name+'_t='+str(t)+'_k='
                         +str(k)+'_id='+str(subsample_id)+'_lr='+str(lr)+'_epoch='+str(num_epoch)+'.json')
            with open(file_path, 'r') as file:
                output = json.load(file)
            ifkm_tilted_sse = output['tilted_SSE_iteration']
            ifkm_tilted_sse_id.append(ifkm_tilted_sse)
        ifkm_tilted_sse_epoch.append([sum(col) / len(col) for col in zip(*ifkm_tilted_sse_id)])
    ifkm_tilted_sse_t[dataset_name] = ifkm_tilted_sse_epoch



print('plot figure...')

def smooth_curve(data, window_size=5):
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, mode='valid')




fig, axs = plt.subplots(2, 4, figsize=(22, 9))

line_styles = ['-', '-', '-', '-', '-', '-', '-']
line_symbols = ['o', 's', 'D', 'v', '*', '^', '+']
fig_name =  ['Athlete','Bank','Census','Diabetes','Recruitment','Spanish','Student','3D-spatial']
dict_in_names = ['TKM, E=1','TKM, E=3', 'TKM, E=5', 'TKM, E=7', 'TKM, E=9']

x = np.arange(0, 500)
linewidth = 2

for i, ax in enumerate(axs.flatten()):
    dataset_name = dataset_names[i]
    ax.plot(smooth_curve(ifkm_tilted_sse_t[dataset_names[i]][0]), line_styles[0], linewidth=linewidth)
    ax.plot(smooth_curve(ifkm_tilted_sse_t[dataset_names[i]][1]), line_styles[1], linewidth=linewidth)
    ax.plot(smooth_curve(ifkm_tilted_sse_t[dataset_names[i]][2]), line_styles[2], linewidth=linewidth)
    ax.plot(smooth_curve(ifkm_tilted_sse_t[dataset_names[i]][3]), line_styles[3], linewidth=linewidth)
    ax.plot(smooth_curve(ifkm_tilted_sse_t[dataset_names[i]][4]), line_styles[4], linewidth=linewidth)

    ax.set_title(fig_name[i], fontsize=18, y=1)
    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Tilted SSE', fontsize=16)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.major.formatter._mathTextSciNotation = True
labels = ['TKM, $E=1$','TKM, $E=3$', 'TKM, $E=5$', 'TKM, $E=7$', 'TKM, E=9']
legend = fig.legend(labels, loc='upper center', ncol=7, fontsize=20, frameon=False)


fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.subplots_adjust(hspace=0.48)
plt.savefig("../results/tilted_sse_epoch.pdf")

plt.show()
