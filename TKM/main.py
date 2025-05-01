import numpy as np
import pandas as pd
from utils import read_data, cluster_variances, setup_seed, compute_tilted_sse_InEachCluster, initialization
from update import tilted_mini_batch_kmeans, FastTKM
from options import args_parser
import json
import time
args = args_parser()
setup_seed(args.seed)
print('random seed =', args.seed)
print('Initialization method: ', args.init)
# csv_file_path = 'data/CleanData/'+args.dataset+'.csv'
num_subsample = args.num_subsample
sample_size = args.sample_size
# dataset_name = args.dataset
dataset_names = ['athlete','bank','census','diabetes','recruitment','spanish','student','3d']
# dataset_names = ['hmda']
for dataset_name in dataset_names:
    for subsample_id in range(0, num_subsample):
        csv_file_path = '../individually-fair-k-clustering-main/data/'+dataset_name+'_'+str(sample_size)+'_'+str(subsample_id)+'.csv'
        data = read_data(csv_file_path)
        t_list = args.t
        k_list = args.num_clusters
        epoch_list = args.epoch_list
        lr_list = args.lr_list
        for k in k_list:
            for t in t_list:
                for num_epoch in epoch_list:
                    for lr in lr_list:
                        output = {}
                        time1 = time.monotonic()
                        centroids_init, labels_init = initialization(data, k, args)
                        centroids, labels, SSE, tilted_SSE = tilted_mini_batch_kmeans(data, args, t, k, num_epoch, lr, centroids_init, labels_init)


                        """ When running FastTKM, please uncomment the following code. """
                        # phi = compute_tilted_sse_InEachCluster(data, centroids, labels, k, t)
                        # centroids, labels, SSE, tilted_SSE = FastTKM(data, args, t, k, num_epoch, lr, centroids_init, labels_init, phi)
                        time2 = time.monotonic()
                        variances, max_distance = cluster_variances(data, centroids, labels)

                        # print("Tilted Mini-Batch K-Means center:\n", centroids)

                        print("SSE in each iteration:\n", SSE)
                        print("tilted SSE in each iteration:\n", tilted_SSE)


                        print(f't={t}, k={k}')
                        for i, variance in enumerate(variances):
                            print(f"Cluster {i+1} Variance: {variance}")
                        print("Additive Variance: ", sum(variances))
                        print('Running time:', time2-time1)

                        output['dataset'] = dataset_name
                        output['SSE_iteration'] = SSE
                        output['tilted_SSE_iteration'] = tilted_SSE
                        output['SSE'] = np.mean(SSE[-20:])/sample_size
                        output['tilted_SSE'] = min(tilted_SSE)
                        output['variances'] = variances
                        output['max_distance'] = max_distance
                        address = 'output/'


                        file_name = (address + dataset_name + '_t=' + str(t) + '_k=' + str(k) + '_id=' +
                                     str(subsample_id) + '_lr=' +str(lr)+'_epoch=' + str(num_epoch) + '.json')
                        with open(file_name, "w") as dataf:
                            json.dump(output, dataf)
