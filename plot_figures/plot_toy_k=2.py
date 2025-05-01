import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from TKM.update import tilted_mini_batch_kmeans
from TKM.options import args_parser
from TKM.utils import read_data, cluster_variances, setup_seed
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({'font.size': 18})
seed = 3
np.random.seed(seed)
X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.3, random_state=0)

noise_points_x1 = np.random.rand(15) * (0.5) + 0.5
noise_points_y1 = np.random.rand(15) * (-1)
noise_points1 = np.column_stack((noise_points_x1, noise_points_y1))
X = np.vstack([X, noise_points1])


noise_points2 = np.random.rand(10, 2) * np.array([0.5, 0.5]) + np.array([2.0, 6])
X = np.vstack([X, noise_points2])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

cluster_colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

plt.figure(figsize=(6, 5))

for cluster_idx in range(kmeans.n_clusters):
    cluster_mask = (y_kmeans == cluster_idx)
    plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], c=cluster_colors[cluster_idx], s=20)

centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1)
args = args_parser()
alpha = 0
point_num = 60
# colors = plt.cm.cool(np.linspace(0, 1, point_num))


colors = [(0, 0, 1), (1, 0, 0)]
cmap_name = 'blue_to_red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=point_num)
i = 0
num_epoch = 5
lr = 0.01
for t in np.logspace(-2, 2, 60):
    print('t='+str(t))
    alpha = alpha + 0.008
    centroids, labels, losses, tilted_SSE = tilted_mini_batch_kmeans(X, args, t, 2, num_epoch, lr)
    variances = cluster_variances(X, centroids, labels)

    plt.scatter(centroids[:, 0], centroids[:, 1], c=cmap(i), s=60, alpha=alpha)
    i = i + 1
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)
plt.title('$k=2$', fontsize=18)
plt.tight_layout()

plt.savefig("k=2_visualization.pdf")
plt.show()
