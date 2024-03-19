import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ifkm_codes.utils import compute_tilted_sse


def kmeans_plusplus_init(X, k):
    centers = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centers]) for x in X])
        prob = distances / distances.sum()
        cumulative_prob = prob.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if r < p:
                centers.append(X[j])
                break
    return np.array(centers)
def tilted_mini_batch_kmeans(X, args, t, k, num_epoch, lr):
    batch_size = args.num_batch
    max_iters = args.maxIter
    n_samples, n_features = X.shape
    if args.init == 'kmeans++':
        centroids = kmeans_plusplus_init(X, k)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
    elif args.init == 'lloyd':
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=1, max_iter=1000, tol=0.02).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    else:
        exit('Error: unrecognized initialization')
    print('Initialization complete...')
    batch_count = n_samples // batch_size
    SSE_all = []
    tilted_SSE_all = []
    for _ in range(max_iters):
        for _ in range(num_epoch):
            if t == 0:
                distances_to_centroids = np.exp(t * np.linalg.norm(X - centroids[labels], axis=1)**2)
                cluster_distances_sum = np.zeros((k,))
            else:
                distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1)**2

            phi = np.zeros((k,))
            for j in range(k):
                if t == 0:
                    cluster_distances_sum[j] = np.sum(distances_to_centroids[labels == j])
                else:
                    phi[j] = (np.logaddexp.reduce(t * distances_to_centroids[labels == j]) + np.log(1/n_samples))/t
            if t == 0:
                weights = (distances_to_centroids / cluster_distances_sum[labels]).reshape(n_samples,1)
            else:
                weights = (np.exp(t * (distances_to_centroids - phi[labels]))/n_samples).reshape(n_samples,1)



            # for i in range(batch_count):
            batch_indices = np.random.choice(n_samples, batch_size, replace=True)

            batch = X[batch_indices]
            weights_batch = weights[batch_indices]
            distances_batch = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
            labels_batch = np.argmin(distances_batch, axis=1)
            gradients = 2 * (batch - centroids[labels_batch])
            weighted_gradients = np.multiply(np.repeat(weights_batch, gradients.shape[1], axis=1), gradients)
            new_centroids = np.array([weighted_gradients[labels_batch == j].sum(axis=0) for j in range(k)])

            # update centroids
            learning_rate = lr
            centroids = centroids + learning_rate * new_centroids

        # compute loss
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        SSE = np.sum((X - centroids[labels]) ** 2)

        if t == 0:
            tilted_SSE = 0
        else:
            tilted_SSE = compute_tilted_sse(X, centroids, labels, k, t, n_samples)


        SSE_all.append(SSE)
        tilted_SSE_all.append(tilted_SSE)
    return centroids, labels, SSE_all, tilted_SSE_all
