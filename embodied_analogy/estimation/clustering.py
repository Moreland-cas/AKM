import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import tracksNd_variance_np, tracksNd_variance_torch
from embodied_analogy.visualization.vis_tracks_2d import vis_tracks2d_napari
from embodied_analogy.visualization.vis_tracks_3d import vis_tracks3d_napari

def cluster_tracks_2d(rgb_seq, tracks_2d, use_diff=True, visualize=False, viewer_title="napari"):
    """
        tracks_2d: torch.tensor([T, M, Nd])
        将 pred_tracks_3d 进行聚类，得到两个部分，分别代表移动和静止的轨迹, 返回两个 mask
    """
    pred_tracks_2d = tracks_2d.cpu()
    T, M, _ = pred_tracks_2d.shape
    
    if use_diff:
        # 减去初始帧, 也就是用 difference 作为 clustering 的输入
        pred_tracks_2d = pred_tracks_2d - pred_tracks_2d[0:1, ...]
        
    motion_for_kmeans = pred_tracks_2d.permute(1, 0, 2).reshape(M, -1) # M T*Nd
    _, moving_labels, _ = cluster.k_means(motion_for_kmeans.numpy(), init="k-means++", n_clusters=2)

    # 确定哪一类是 moving, 那一类是 static, 依据是轨迹的 variance
    part1 = pred_tracks_2d[:, moving_labels == 0, :] # T, M1, Nd
    part2 = pred_tracks_2d[:, moving_labels == 1, :] # T, M2, Nd

    if tracksNd_variance_torch(part1) > tracksNd_variance_torch(part2):
        moving_mask = (moving_labels == 0)
        static_mask = (moving_labels == 1)
    else:
        moving_mask = (moving_labels == 1)
        static_mask = (moving_labels == 0)

    if visualize:
        rigid_part_colors = np.zeros((tracks_2d.shape[1], 3)) # M, 3
        rigid_part_colors[moving_mask] = np.array([1, 0, 0]) # red
        rigid_part_colors[static_mask] = np.array([0, 0, 1]) # blue
        vis_tracks2d_napari(rgb_seq, tracks_2d, colors=rigid_part_colors, viewer_title=viewer_title)
        
    return moving_mask, static_mask


def cluster_tracks_3d(tracks_3d, use_diff=True, visualize=False, viewer_title="napari"):
    """
        tracks_3d: np.array([T, M, Nd])
        将 tracks_3d 进行聚类，得到两个部分，分别代表移动和静止的轨迹, 返回两个 mask
    """
    tracks_3d_ = np.copy(tracks_3d)
    T, M, _ = tracks_3d_.shape
    
    if use_diff:
        # 减去初始帧, 也就是用 difference 作为 clustering 的输入
        tracks_3d_ = tracks_3d_ - tracks_3d_[0:1, ...]
        
    motion_for_kmeans = tracks_3d_.transpose(1, 0, 2).reshape(M, -1) # M T*Nd
    _, moving_labels, _ = cluster.k_means(motion_for_kmeans, init="k-means++", n_clusters=2)

    # 确定哪一类是 moving, 那一类是 static, 依据是轨迹的 variance
    part1 = tracks_3d_[:, moving_labels == 0, :] # T, M1, Nd
    part2 = tracks_3d_[:, moving_labels == 1, :] # T, M2, Nd

    if tracksNd_variance_np(part1) > tracksNd_variance_np(part2):
        moving_mask = (moving_labels == 0)
        static_mask = (moving_labels == 1)
    else:
        moving_mask = (moving_labels == 1)
        static_mask = (moving_labels == 0)

    if visualize:
        rigid_part_colors = np.zeros((tracks_3d.shape[1], 3)) # M, 3
        rigid_part_colors[moving_mask] = np.array([1, 0, 0]) # red
        rigid_part_colors[static_mask] = np.array([0, 0, 1])
        vis_tracks3d_napari(tracks_3d, colors=rigid_part_colors, viewer_title=viewer_title)
        
    return moving_mask, static_mask