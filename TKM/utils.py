import pandas as pd
import numpy as np
# import torch
import random

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

# compute variance in each cluster
def cluster_variances(X, centroids, labels):
    variances = []
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances)
        variances.append(variance)
    return variances

def compute_tilted_sse(X, centroids, labels, k, t, n_samples):
    distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1) ** 2
    phi = np.zeros((k,))
    for j in range(k):
        phi[j] = (np.logaddexp.reduce(t * distances_to_centroids[labels == j]) + np.log(1 / n_samples)) / t
    return sum(phi)
def exp_details(args):
    print('     Running ' + args.alg + 'on ' + args.dataset )
    print('\nParameter description')
    print(f'    Number of clusters : {args.num_clusters}')
    print(f'    Epoch size         : {args.num_epoch}')
    print(f'    Batch size         : {args.num_batch}')
    print(f'    Learning rate      : {args.lr}\n')
    print(f'    Maximum iterations : {args.maxIter}\n')

    return