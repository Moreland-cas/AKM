"""
Input:
    rgb_seq: [t, h, w]
    depth_seq: [t, h, w]
    object_mask: [t, h, w]
    franka_mask: [t, h, w]
Output:
    moving_mask: [t, h, w]
    joint_states: [t, 1]
    joint_params:
        (3,) for prismatic joint
        (6,) for revolute joint
"""
import torch
import numpy as np
from PIL import Image
import sklearn.cluster as cluster
from cotracker.utils.visualizer import Visualizer
from embodied_analogy.process_record import RecordDataReader
from embodied_analogy.online_cotracker import track_any_points
from embodied_analogy.utils import (
    depth_image_to_pointcloud, 
    visualize_pc, 
    camera_to_image,
    draw_points_on_image
)

def filter_by_visibility(pred_tracks, pred_visibility):
    """
        pred_tracks: [T, M, 2]
        pred_visibility: [T, M]
    """
    always_visible_mask = pred_visibility.all(dim=0) # num_clusters
    pred_tracks = pred_tracks[:, always_visible_mask, :] # T M_ 2
    return pred_tracks

def filter_by_sam2(pred_tracks, sam2_mask):
    # TODO
    return pred_tracks

def extract_tracked_depths(depth_seq, pred_tracks):
    """
    Args:
        depth_seq (np.ndarray): 
            深度图视频，形状为 (T, H, W, 1)
        pred_tracks (torch.Tensor): 
            (T, M, 2)
            每个点的坐标形式为 [u, v]，值域为 [0, W) 和 [0, H)。
        
    返回:
        torch.Tensor: 点的深度值，形状为 (T, M)，每个点的深度值。
    """
    depth_seq = depth_seq.squeeze(-1)
    T, H, W = depth_seq.shape
    # _, M, _ = pred_tracks.shape

    # 确保 pred_tracks 是整数坐标
    u_coords = pred_tracks[..., 0].clamp(0, W - 1).long()  # 水平坐标 u
    v_coords = pred_tracks[..., 1].clamp(0, H - 1).long()  # 垂直坐标 v

    # 将 depth_seq 转为 torch.Tensor
    depth_tensor = torch.from_numpy(depth_seq).cuda()  # Shape: (T, H, W)

    # 使用高级索引提取深度值
    depth_tracks = depth_tensor[torch.arange(T).unsqueeze(1), v_coords, u_coords]  # Shape: (T, M)
    return depth_tracks.cpu().numpy()


def filter_by_depthSeq(pred_tracks, depth_tracks, thr=1.5):
    """
    过滤深度序列中的无效序列，基于相邻元素的相对变化率。
    
    参数:
        pred_tracks (torch.Tensor): 
            形状为 (T, M, 2)，每个点的坐标形式为 [u, v]，值域为 [0, W) 和 [0, H)。
        depth_tracks (np.array): 
            形状为 (T, N)，表示 T 个时间步上的 N 个深度序列。
        thr (float): 
            相对变化率的阈值，超过此比率的序列会被判定为无效。
    
    返回:
        torch.Tensor: 
            筛选后的 pred_tracks, 形状为 (T, M', 2)，其中 M' 是筛选后的有效点数量。
    """
    depth_tracks += 1e-6  # 防止除零
    # 转置 depth_tracks，使其形状为 (N, T)
    depth_tracks = depth_tracks.T  # (N, T)
    
    # 计算相邻元素的变化比率 (N, T-1)
    ratios = np.maximum(depth_tracks[:, :-1] / depth_tracks[:, 1:], depth_tracks[:, 1:] / depth_tracks[:, :-1])
    
    # 检测每个序列是否存在无效的变化
    invalid_sequences = np.any(ratios > thr, axis=1)  # 形状为 (N,)
    
    # 合法序列的掩码
    valid_mask = ~invalid_sequences  # 形状为 (N,)
    
    # 筛选 pred_tracks 的合法序列
    pred_tracks = pred_tracks[:, valid_mask, :]  # 筛选有效的 M 点
    
    return pred_tracks

def visualize_tracks_on_video(rgb_frames, pred_tracks, file_name="video_output"):
    """
        Args:
            rgb_frames: [T, H, W, 3]
            pred_tracks: [T, M, 2]
    """
    video = torch.tensor(np.stack(rgb_frames), device="cuda").permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./", pad_value=120, linewidth=1)
    
    pred_tracks = pred_tracks[None] # [1, T, M, 2]
    # pred_visibility = torch.ones((pred_tracks.shape[:3], 1), dtype=torch.bool, device="cuda")
    vis.visualize(video, pred_tracks, filename=file_name)
    
    
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-01-07_18-06-10.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = dr.depth # T H W 1
K = dr.intrinsic # 3, 3
object_mask_0 = dr.seg
visualize = True

pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255. # N,3
if visualize:
    visualize_pc(pc_0, rgb_0)

# 对初始帧中的物体点云聚类得到 num_clusters 个点簇, 并计算 num_clusters 个簇中心在图像上的投影
# TODO: 可以在一开始聚类的时候也考虑视觉特征
num_clusters = 300
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centroids, labels, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

color_map = np.random.rand(num_clusters, 3)
colors = color_map[labels]

if visualize:
    visualize_pc(pc_0, colors)

# 然后对这 k 个点进行 rgb tracking, 筛选出稳定的 m 个点
centroids_camera = centroids[:, :3] # num_clusters, 3
centroids_image = camera_to_image(centroids_camera, K) # num_clusters, 2

if visualize:
    draw_points_on_image(Image.fromarray(rgb_seq[0]), centroids_image).show()

# [T, M, 2], [T, M]
pred_tracks, pred_visibility = track_any_points(rgb_seq, centroids_image, visiualize=visualize)

# 对跟踪的点进行筛选
# 1）要是一直能跟踪到的点
pred_tracks = filter_by_visibility(pred_tracks, pred_visibility)

if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks, "after_vis_filter")
    
# 2）要是一直在物体区域内的点(物体区域由sam2得到)
pred_tracks = filter_by_sam2(pred_tracks, sam2_mask=None)

if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks, "after_sam2_filter")
    
# 3）这些点的深度不应该有突然的变化
# TODO: 改为线性模型的回归
depth_tracks = extract_tracked_depths(depth_seq, pred_tracks) # B, M
pred_tracks = filter_by_depthSeq(pred_tracks, depth_tracks) # B, M, 2

if visualize:
    visualize_tracks_on_video(rgb_seq, pred_tracks, "after_depth_filter")

# 根据这 m 个点的轨迹将其聚类为 static 和 moving 两类
a = 1
# 对于剩下的 k-m 个点簇进行分类验证, 分为 static, moving 和 unknown 三类