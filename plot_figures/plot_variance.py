import json
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

print('loading data...')
k_num = [3,4,5,6,7,8,9,10]
num_subsample = 10

dataset_names = ['athlete','bank','census','diabetes','recruitment','spanish','student','3d']
sse_dict = {}
variance_dict = {}
for dataset_name in dataset_names:
    jkl_variances_k, mv_variances_k, fr_variances_k, sfr_variance_k = [], [], [], []
    jkl_sse_k, mv_sse_k, fr_sse_k, sfr_sse_k = [], [], [], []
    sse_dict_k, variance_dict_k = {}, {}
    for k in k_num:
        jkl_variances, mv_variances, fr_variances, sfr_variances = [], [], [], []
        jkl_sse, mv_sse, fr_sse, sfr_sse = [], [], [], []
        for subsample_id in range(num_subsample):
            file_path = '../individually-fair-k-clustering-main/output/kmeans/'+dataset_name+'_1000_'+str(subsample_id)+'_k_'+str(k)+'.json'
            with open(file_path, 'r') as file:
                output = json.load(file)
                jkl_variance = sorted(output['jkl_output']['variances'], reverse=True)
                mv_variance = sorted(output['mv_output']['variances'], reverse=True)
                fr_variance = sorted(output['fr_output']['variances'], reverse=True)
                sfr_variance = sorted(output['spa_fr_output']['variances'], reverse=True)


                jkl_cost = output['jkl_output']['cost']
                mv_cost = output['mv_output']['cost']
                fr_cost = output['fr_output']['cost']
                sfr_cost = output['spa_fr_output']['cost']
            jkl_variances.append(jkl_variance)
            mv_variances.append(mv_variance)
            fr_variances.append(fr_variance)
            sfr_variances.append(sfr_variance)


            jkl_sse.append(jkl_cost)
            mv_sse.append(mv_cost)
            fr_sse.append(fr_cost)
            sfr_sse.append(sfr_cost)

        jkl_variances_k.append([sum(col) / len(col) for col in zip(*jkl_variances)])
        mv_variances_k.append([sum(col) / len(col) for col in zip(*mv_variances)])
        fr_variances_k.append([sum(col) / len(col) for col in zip(*fr_variances)])
        sfr_variance_k.append([sum(col) / len(col) for col in zip(*sfr_variances)])

        jkl_sse_k.append(sum(jkl_sse) / len(jkl_sse))
        mv_sse_k.append(sum(mv_sse) / len(mv_sse))
        fr_sse_k.append(sum(fr_sse) / len(fr_sse))
        sfr_sse_k.append(sum(sfr_sse) / len(sfr_sse))
    sse_dict_k['jkl_sse'] = jkl_sse_k
    sse_dict_k['mv_sse'] = mv_sse_k
    sse_dict_k['fr_sse'] = fr_sse_k
    sse_dict_k['sfr_sse'] = sfr_sse_k

    variance_dict_k['jkl_variance'] = jkl_variances_k
    variance_dict_k['mv_variance'] = mv_variances_k
    variance_dict_k['fr_variance'] = fr_variances_k
    variance_dict_k['sfr_variance'] = sfr_variance_k


    sse_dict[dataset_name] = sse_dict_k
    variance_dict[dataset_name] = variance_dict_k



t_list = [0,0.01,0.05,0.1, 0.2]



lr = 0.05
num_epoch = 5
for dataset_name in dataset_names:
    for t in t_list:
        ifkm_output_sse = {}
        ifkm_output_variance = {}
        ifkm_sse_k, ifkm_variance_k = [], []
        for k in k_num:
            ifkm_sse_id, ifkm_variance_id = [], []
            for subsample_id in range(num_subsample):
                file_path = '../ifkm_codes/output/'+dataset_name+'_t='+str(t)+'_k='+str(k)+'_id='+str(subsample_id)+ '_lr=' +str(lr)+'_epoch=' + str(num_epoch) +'.json'
                with open(file_path, 'r') as file:
                    output = json.load(file)
                ifkm_sse = output['SSE']
                ifkm_variance = sorted(output['variances'], reverse=True)

                ifkm_sse_id.append(ifkm_sse)
                ifkm_variance_id.append(ifkm_variance)

            ifkm_variance_k.append([sum(col) / len(col) for col in zip(*ifkm_variance_id)])
            ifkm_sse_k.append(sum(ifkm_sse_id) / len(ifkm_sse_id))

        sse_dict[dataset_name]['ifkm, t='+str(t)] = ifkm_sse_k
        variance_dict[dataset_name]['ifkm, t='+str(t)] = ifkm_variance_k

print('plot figure...')
x = np.array([1.4, 3.6, 5.8, 8.0])
k = 4

dataset_names = ['athlete','bank','census','diabetes','recruitment','spanish','student','3d']
show_names = ['Athlete','Bank','Census','Diabetes','Recruitment','Spanish','Student','3D-spatial']
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 9))
method_names = ['jkl_variance', 'mv_variance', 'fr_variance', 'sfr_variance', 'ifkm, t=0', 'ifkm, t=0.01', 'ifkm, t=0.05', 'ifkm, t=0.1','ifkm, t=0.2']
label_name = ['JKL', 'MV', 'FR', 'SFR', 'NF', 'TKM, $t=0.01$', 'TKM, $t=0.05$', 'TKM, $t=0.1$','TKM, $t=0.2$']
for i, ax in enumerate(axes.flat):
    for j in range(0,9):
        name = dataset_names[i]
        jkl_variance = variance_dict[name]['jkl_variance'][k-3]
        mv_variance = variance_dict[name]['mv_variance'][k-3]
        fr_variance = variance_dict[name]['fr_variance'][k-3]
        sfr_variance = variance_dict[name]['sfr_variance'][k-3]
        ifkm_t_0 = variance_dict[name]['ifkm, t=0'][k - 3]
        ifkm_t_001 = variance_dict[name]['ifkm, t=0.01'][k - 3]
        ifkm_t_01 = variance_dict[name]['ifkm, t=0.05'][k - 3]
        ifkm_t_05 = variance_dict[name]['ifkm, t=0.1'][k - 3]
        ifkm_t_1 = variance_dict[name]['ifkm, t=0.2'][k - 3]
        y_group = np.array([jkl_variance,mv_variance,fr_variance,sfr_variance,ifkm_t_0,ifkm_t_001,ifkm_t_01,ifkm_t_05,ifkm_t_1])



        hatch_styles = ['//', '..', 'xx', '\\\\', '--', '||', '++', 'o','/']
        bars = ax.bar(x + j * 0.2, y_group[j, :], width=0.2, hatch=hatch_styles[j],label=label_name[j])

        show_name = show_names[i]
        ax.set_title(show_name, fontsize=16, y=1)
        ax.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], va='baseline', fontsize=14)
        ax.set_ylabel('Variance', fontsize=16)
        ax.tick_params(axis='x', which='both', pad=24)
        ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1','1.2','1.4'], fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)

        color_indices = [0, 1]

        for a, label in enumerate(ax.get_xticklabels()):
            if a in color_indices:
                label.set_color('black')


    ax.set_xticks(x + 0.8)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=9, fontsize=20, frameon=False)

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.subplots_adjust(hspace=0.3)
plt.savefig("../results/variance.pdf")

plt.show()

