import napari
import numpy as np
import sklearn.cluster as cluster
from sklearn.cluster import SpectralClustering
from embodied_analogy.utility.utils import (
    tracksNd_variance_np, 
    tracksNd_variance_torch,
    napari_time_series_transform
)
from embodied_analogy.utility.utils import vis_tracks2d_napari

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


def cluster_tracks_3d_kmeans(tracks_3d, use_diff=True, visualize=False):
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
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "clustring 3d tracks (kmeas++)"
        
        # 解决 napari 坐标系显示
        napari_tracks_3d = np.copy(tracks_3d)
        napari_tracks_3d[..., -1] *= -1
        viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, moving_mask]), size=0.01, name='moving part', opacity=0.8, face_color="red")
        viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, static_mask]), size=0.01, name='static part', opacity=0.8, face_color="green")
        
        # viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, moving_mask]), size=0.01, name='moving part', opacity=0.8, face_color="red")
        # viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, static_mask]), size=0.01, name='static part', opacity=0.8, face_color="green")
        
        napari.run()
        
    return moving_mask, static_mask


def cluster_tracks_3d_spectral(tracks_3d, feat_type="diff", visualize=False):
    """
    tracks_3d: np.array([T, M, Nd])
    将 tracks_3d 进行聚类，得到两个部分，分别代表移动和静止的轨迹, 返回两个 mask
    """
    assert feat_type in ["diff", "dir"]
    
    tracks_3d_ = np.copy(tracks_3d)
    T, M, _ = tracks_3d_.shape
    
    # 减去初始帧, 也就是用 difference 作为 clustering 的输入
    tracks_3d_diff = tracks_3d_ - tracks_3d_[0:1, ...] # T, M, 3
    tracks_3d_norm = np.linalg.norm(tracks_3d_diff, axis=-1, keepdims=True) # T, M, 1
    
    # 仅对 norms > 0.01 的地方进行归一化; 对于 norms <= 0.01 的地方，保持原本的数据不变
    mask = (tracks_3d_norm > 0.001).astype(np.float32) # 1mm
    tracks_3d_dir_unstable = tracks_3d_diff / np.maximum(tracks_3d_norm, 1e-6)
    tracks_3d_dir = tracks_3d_dir_unstable * mask + (1 - mask) * tracks_3d_diff
        
    # 将数据重塑为 (M, T * Nd)
    diff_feat = tracks_3d_diff.transpose(1, 0, 2).reshape(M, -1)
    dir_feat = tracks_3d_dir.transpose(1, 0, 2).reshape(M, -1)
    
    # TODO 这里也许可以做个 ablation study
    if feat_type == "diff":
        feat_for_spectral = diff_feat
    elif feat_type == "dir":
        feat_for_spectral = dir_feat

    # 使用谱聚类
    n_clusters = 2
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='nearest_neighbors', 
        n_neighbors=10
    )
    spectral_clustering.fit(feat_for_spectral)
    moving_labels = spectral_clustering.labels_

    # 确定哪一类是 moving, 哪一类是 static, 依据是轨迹的 variance
    part1 = tracks_3d_[:, moving_labels == 0, :]  # T, M1, Nd
    part2 = tracks_3d_[:, moving_labels == 1, :]  # T, M2, Nd

    if tracksNd_variance_np(part1) > tracksNd_variance_np(part2):
        moving_mask = (moving_labels == 0)
        static_mask = (moving_labels == 1)
    else:
        moving_mask = (moving_labels == 1)
        static_mask = (moving_labels == 0)

    if visualize:
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "cluster 3d tracks (spectral clustering)"
        
        # 解决 napari 坐标系显示
        napari_tracks_3d = np.copy(tracks_3d)
        napari_tracks_3d[..., -1] *= -1
        viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, moving_mask]), size=0.01, name='moving part', opacity=0.8, face_color="red")
        viewer.add_points(napari_time_series_transform(napari_tracks_3d[:, static_mask]), size=0.01, name='static part', opacity=0.8, face_color="green")
        
        napari.run()
        
    return moving_mask, static_mask