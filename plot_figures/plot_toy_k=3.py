import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from ifkm_codes.update import tilted_mini_batch_kmeans
from ifkm_codes.options import args_parser
from ifkm_codes.utils import read_data, cluster_variances, setup_seed
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 18})

seed = 3    #3
np.random.seed(seed)
# 生成一些示例数据
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
noise_points_x1 = np.random.rand(10) * 0.5 - 2  # [-1.5, -2]
noise_points_y1 = np.random.rand(10) * 0.5 - 1  # [-1.5, -1]
noise_points1 = np.column_stack((noise_points_x1, noise_points_y1))
X = np.vstack([X, noise_points1])
#
noise_points_x2 = np.random.rand(10) * 0.5 + 1  # [1.0, 1.5]
noise_points_y2 = np.random.rand(10) * 0.5 + 1.5 # [1.5, 2]
noise_points2 = np.column_stack((noise_points_x2, noise_points_y2))
X = np.vstack([X, noise_points2])
#
noise_points_x3 = np.random.rand(10) * 0.5 + 1.5  # [1.5, 2]
noise_points_y3 = np.random.rand(10) * 0.5 - 2 # [-1.5, -1]
noise_points3 = np.column_stack((noise_points_x3, noise_points_y3))
X = np.vstack([X, noise_points3])
# 使用K均值聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


cluster_colors = ['#2ca02c', '#ff7f0e', '#1f77b4']  # 例如，红色、绿色、蓝色、黄色
plt.figure(figsize=(6, 5))
# 可视化聚类结果
for cluster_idx in range(kmeans.n_clusters):
    cluster_mask = (y_kmeans == cluster_idx)
    plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], c=cluster_colors[cluster_idx], s=20)


#
# # 可视化聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis')

# 绘制聚类中心
centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1)
args = args_parser()
alpha = 0
point_num = 60
# colors = plt.cm.cool(np.linspace(0, 1, point_num))

colors = [(0, 0, 1), (1, 0, 0)]  # 蓝色到红色
cmap_name = 'blue_to_red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=point_num)
i = 0
for t in np.logspace(-2, 2, point_num):
    print('t='+str(t))
    centroids, labels, losses, _ = tilted_mini_batch_kmeans(X, args, t,3, num_epoch=5, lr=0.01)
    variances = cluster_variances(X, centroids, labels)

    plt.scatter(centroids[:, 0], centroids[:, 1], c=cmap(i), s=60, alpha=1)
    i = i+1
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)
plt.title('$k=3$', fontsize=18)


plt.tight_layout()
plt.savefig("k=3_visualization.pdf")
plt.show()
