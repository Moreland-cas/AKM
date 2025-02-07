import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import tracksNd_variance

def cluster_tracks_Nd(pred_tracks_Nd):
    """
        pred_tracks_Nd: torch.tensor([T, M, Nd])
        将 pred_tracks_Nd 进行聚类，得到两个部分，分别代表移动和静止的轨迹, 返回两个 mask
    """
    pred_tracks_Nd = pred_tracks_Nd.cpu()
    T, M, _ = pred_tracks_Nd.shape
    motion_for_kmeans = pred_tracks_Nd.permute(1, 0, 2).reshape(M, -1) # M T*Nd
    _, moving_labels, _ = cluster.k_means(motion_for_kmeans.numpy(), init="k-means++", n_clusters=2)

    # 确定哪一类是 moving, 那一类是 static, 依据是轨迹的 variance
    part1 = pred_tracks_Nd[:, moving_labels == 0, :] # T, M1, Nd
    part2 = pred_tracks_Nd[:, moving_labels == 1, :] # T, M2, Nd

    if tracksNd_variance(part1) > tracksNd_variance(part2):
        moving_mask = (moving_labels == 0)
        static_mask = (moving_labels == 1)
    else:
        moving_mask = (moving_labels == 1)
        static_mask = (moving_labels == 0)

    return moving_mask, static_mask