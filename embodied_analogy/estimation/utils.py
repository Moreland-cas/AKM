import os
import torch
import numpy as np
from scipy.spatial import cKDTree
from cotracker.utils.visualizer import Visualizer
from embodied_analogy.utility.constants import *
from embodied_analogy.visualization import *


def extract_tracked_depths(depth_seq, pred_tracks):
    """
    Args:
        depth_seq (np.ndarray): 
            深度图视频,形状为 (T, H, W)
        pred_tracks (torch.Tensor): 
            (T, M, 2)
            每个点的坐标形式为 [u, v],值域为 [0, W) 和 [0, H)。
        
    返回:
        torch.Tensor: 点的深度值,形状为 (T, M),每个点的深度值。
    """
    T, H, W = depth_seq.shape
    # _, M, _ = pred_tracks.shape

    # 确保 pred_tracks 是整数坐标
    u_coords = pred_tracks[..., 0].clamp(0, W - 1).long()  # 水平坐标 u
    v_coords = pred_tracks[..., 1].clamp(0, H - 1).long()  # 垂直坐标 v

    # 将 depth_seq 转为 torch.Tensor
    depth_tensor = torch.from_numpy(depth_seq).cuda()  # Shape: (T, H, W)

    # 使用高级索引提取深度值
    depth_tracks = depth_tensor[torch.arange(T).unsqueeze(1), v_coords, u_coords]  # Shape: (T, M)
    return depth_tracks.cpu()

def filter_tracks2d_by_visibility(rgb_seq, pred_tracks_2d, pred_visibility, visualize=False):
    """
        pred_tracks_2d: torch.tensor([T, M, 2])
        pred_visibility: torch.tensor([T, M])
    """
    always_visible_mask = pred_visibility.all(dim=0) # num_clusters
    pred_tracks_2d = pred_tracks_2d[:, always_visible_mask, :] # T M_ 2
    
    if visualize:
        vis_tracks2d_napari(rgb_seq, pred_tracks_2d, viewer_title="filter_tracks2d_by_visibility")
    return pred_tracks_2d

def filter_tracks2d_by_depthSeq_mask(rgb_seq, pred_tracks_2d, depthSeq_mask, visualize=False):
    """
    筛选出那些不曾落到 invalid depth_mask 中的 tracks
    Args:
        pred_tracks_2d: torch.tensor([T, M, 2])
        depthSeq_mask: np.array([T, H, W], dtype=np.bool_)
    """
    T = pred_tracks_2d.shape[0]
    # 首先读取出 pred_tracks_2d 在对应位置的 mask 的值
    pred_tracks_2d_floor = torch.floor(pred_tracks_2d).long() # T M 2 (u, v)
    t_index = torch.arange(T, device=pred_tracks_2d.device).view(T, 1)  # (T, M)
    pred_tracks_2d_mask = depthSeq_mask[t_index, pred_tracks_2d_floor[:, :, 1], pred_tracks_2d_floor[:, :, 0]] # T M
    
    # 然后在时间维度上取交, 得到一个大小为 M 的 mask, 把其中一直为 True 的保留下来并返回
    mask_and = np.all(pred_tracks_2d_mask, axis=0) # M
    
    pred_tracks_2d = pred_tracks_2d[:, mask_and, :]
    if visualize:
        vis_tracks2d_napari(rgb_seq, pred_tracks_2d, viewer_title="filter_tracks2d_by_depthSeq_mask")
    return pred_tracks_2d

def filter_tracks2d_by_depthSeq_diff(pred_tracks, depth_tracks, thr=1.5):
    """
    过滤深度序列中的无效序列,基于相邻元素的相对变化率。
    
    参数:
        pred_tracks (torch.Tensor): 
            形状为 (T, N, 2),每个点的坐标形式为 [u, v],值域为 [0, W) 和 [0, H)。
        depth_tracks (torch.Tensor): 
            形状为 (T, N),表示 T 个时间步上的 N 个深度序列。
        thr (float): 
            相对变化率的阈值,超过此比率的序列会被判定为无效。
    
    返回:
        torch.Tensor: 
            筛选后的 pred_tracks, 形状为 (T, M, 2),其中 M 是筛选后的有效点数量。
    """
    depth_tracks += 1e-6  # 防止除零

    # 转置 depth_tracks,使其形状为 (N, T)
    depth_tracks_T = depth_tracks.T  # (N, T)

    # 计算相邻元素的变化比率 (N, T-1)
    ratios = torch.max(depth_tracks_T[:, :-1] / depth_tracks_T[:, 1:], depth_tracks_T[:, 1:] / depth_tracks_T[:, :-1])

    # 检测每个序列是否存在无效的变化 (N,)
    invalid_sequences = torch.any(ratios > thr, dim=1)  # 检查每一行是否有任何值大于阈值

    # 合法序列的掩码 (N,)
    valid_mask = ~invalid_sequences  # 合法序列为 False,其他为 True
    
    # 筛选 pred_tracks 和 depth_tracks 的合法序列
    pred_tracks = pred_tracks[:, valid_mask, :]  # 筛选有效的 M 点,(T, M, 2)
    depth_tracks = depth_tracks[:, valid_mask]   # 筛选有效的 M 深度,(T, M)
    
    return pred_tracks, depth_tracks

def visualize_tracks2d_on_video(rgb_frames, pred_tracks, file_name="video_output", vis_folder="./"):
    """
        Args:
            rgb_frames: np.array([T, H, W, 3])
            pred_tracks: torch.Tensor([T, M, 2])
    """
    os.makedirs(vis_folder, exist_ok=True)
    video = torch.tensor(np.stack(rgb_frames), device="cuda").permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir=vis_folder, pad_value=120, linewidth=1)
    
    pred_tracks = pred_tracks[None] # [1, T, M, 2]
    # pred_visibility = torch.ones((pred_tracks.shape[:3], 1), dtype=torch.bool, device="cuda")
    vis.visualize(video, pred_tracks, filename=file_name)


def get_dynamic_seg(mask, moving_points, static_points, visualize=False):
    """
    根据A和B点集的最近邻分类让mask中为True的点分类。

    参数:
        mask (np.ndarray): 大小为(H, W)的应用匹配网络,mask=True的点需要分类
        points_A (np.ndarray): 大小为(M, 2)的A类点集,具有(u, v)坐标
        points_B (np.ndarray): 大小为(N, 2)的B类点集,具有(u, v)坐标

    返回:
        dynamic_mask (np.ndarray): 大小为(H, W)的分类结果,A类标记1,B类标记2
    """
    H, W = mask.shape

    # 构建KD树以加快最近邻搜索，确保用(v, u)坐标格式
    tree_A = cKDTree(moving_points[:, [1, 0]])
    tree_B = cKDTree(static_points[:, [1, 0]])

    # 找到mask中为True的点坐标
    mask_indices = np.argwhere(mask) # N, 2 (v, u)

    # 对每个True点计算自A和B集合的最近距离
    distances_A, _ = tree_A.query(mask_indices)
    distances_B, _ = tree_B.query(mask_indices)

    # 初始化结果网络
    dynamic_mask = np.zeros((H, W), dtype=np.uint8)

    # 根据最近邻距离分类
    A_closer = distances_A < distances_B # N
    B_closer = ~A_closer

    # 将分类结果填入到对应位置
    dynamic_mask[mask_indices[A_closer, 0], mask_indices[A_closer, 1]] = MOVING_LABEL
    dynamic_mask[mask_indices[B_closer, 0], mask_indices[B_closer, 1]] = STATIC_LABEL

    if visualize:
        import napari 
        viewer = napari.view_image(mask, rgb=True)
        viewer.add_labels(mask.astype(np.int32), name='articulated objects')
        viewer.add_labels((dynamic_mask == MOVING_LABEL).astype(np.int32) * 2, name='moving parts')
        viewer.add_labels((dynamic_mask == STATIC_LABEL).astype(np.int32) * 3, name='static parts')
        napari.run()
    return dynamic_mask

def get_dynamic_seg_seq(mask_seq, moving_points_seq, static_points_seq, visualize=False):
    T = mask_seq.shape[0]
    dynamic_seg_seq = []
    
    for i in range(T):
        dynamic_seg = get_dynamic_seg(mask_seq[i], moving_points_seq[i], static_points_seq[i])
        dynamic_seg_seq.append(dynamic_seg)
    
    dynamic_seg_seq = np.array(dynamic_seg_seq)
    
    if visualize:
        import napari 
        viewer = napari.view_image(mask_seq, rgb=False)
        viewer.title = "dynamic segment video mask by tracks2d"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels((dynamic_seg_seq == MOVING_LABEL).astype(np.int32) * 2, name='moving parts')
        viewer.add_labels((dynamic_seg_seq == STATIC_LABEL).astype(np.int32) * 3, name='static parts')
        napari.run()
    return dynamic_seg_seq # T, H, W


def quantile_sampling(arr, M):
    """
    从数组中按分位数抽样，返回代表性数据点。
    
    参数:
    arr (np.ndarray): 输入的一维数据数组。
    M (int): 要抽取的数据点数量。

    返回:
    np.ndarray: 抽取的代表性数据点。
    """
    # 确保输入为NumPy数组并排序
    arr_sorted = np.sort(arr)
    
    # 计算分位数对应的索引
    quantiles = np.linspace(0, 1, M)
    indices = (quantiles * (len(arr_sorted) - 1)).astype(int)
    
    # 根据索引抽取数据
    sampled_data = arr_sorted[indices]
    return sampled_data

