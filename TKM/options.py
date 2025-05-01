import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='3d', choices=['bank', 'census', 'diabetes','3d'], help="dataset name")
    parser.add_argument('--num_clusters', type=list, default=[4], help="number of clusters, k in our paper")
    parser.add_argument('--num_subsample', type=int, default=10, help="subsample number of the full dataset")
    parser.add_argument('--sample_size', type=int, default=1000, help="subsample size of the dataset")
    parser.add_argument('--init', type=str, default='lloyd', choices=['kmeans++','random', 'lloyd'], help="initialization method")
    parser.add_argument('--t', type=list, default=[0.1], help="t in our paper, when t=0, tkm generalize to kmeans via sgd")
    parser.add_argument('--maxIter', type=int, default=500, help="maximum iterations")
    parser.add_argument('--epoch_list', type=list, default=[5], help="number of local epochs: E")
    parser.add_argument('--num_batch', type=int, default=50, help="local batch size: B")
    parser.add_argument('--lr_list', type=float, default=[0.05], help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    args = parser.parse_args()
    return args