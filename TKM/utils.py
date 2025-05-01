import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data.values



def compute_tilted_sse(X, centroids, labels, k, t, n_samples):
    distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1) ** 2
    phi = np.zeros((k,))
    for j in range(k):
        phi[j] = (np.logaddexp.reduce(t * distances_to_centroids*((labels == j).astype(int))) + np.log(1/n_samples))/t
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


def initialization(X, k, args):
    """
    Randomly initialize K-means cluster centroids and assign sample labels.

    Parameters:
    data: Input data, shape (n_samples, n_features)
    k: Number of clusters

    Returns:
    centroids: Initialized centroids
    labels: Labels for each sample
    """

    if args.init == 'random':
        # Randomly select k samples as initial centroids
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        # Calculate the distance from each sample to each centroid and assign labels
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
    elif args.init == 'kmeans++':
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=1, max_iter=1000, tol=0.02).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    elif args.init == 'lloyd':
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=1, max_iter=1000, tol=0.02).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    else:
        exit('Error: unrecognized initialization')
    return centroids, labels



def compute_tilted_sse_InEachCluster(X, centroids, labels, k, t):
    distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1) ** 2
    phi = np.zeros((k,))
    n_samples = X.shape[0]
    for j in range(k):
        phi[j] = (np.logaddexp.reduce(t * distances_to_centroids*((labels == j).astype(int))) + np.log(1/n_samples))/t
    return phi


def cluster_variance(X, centroids, labels):
    variances, largest_dist = [], []

    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances)
        variances.append(variance)
        if distances.size == 0:
            largest_dist.append(0)
        else:
            largest_dist.append(max(distances))
    return variances, largest_dist


