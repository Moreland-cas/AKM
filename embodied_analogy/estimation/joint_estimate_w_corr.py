"""
Input:
    T, M, 3 的点云轨迹
    
Output:
    输出平移关节参数和误差统计
    输出旋转关节参数和误差统计

"""
import torch
import numpy as np
from scipy.linalg import svd


def estimate_translation_from_tracks(tracks_3d):
    """
    通过所有时间帧的位移变化估计平移方向，并计算每帧沿该方向的位移标量
    :param tracks_3d: 形状为(T, M, 3)的numpy数组, T是时间步数, M是点的数量
    :return: 平移方向的平均单位向量 (3,), 每帧的位移标量数组 (T,)
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
        
    T, M, _ = tracks_3d.shape
    unit_vectors = []  # 用于存储所有时间帧间的单位向量

    # 遍历所有帧（从t2到tN，与t1的差异）
    for t in range(1, T):
        displacement_vectors = tracks_3d[t] - tracks_3d[0]  # 当前帧和初始帧的位移
        norms = np.linalg.norm(displacement_vectors, axis=1, keepdims=True)  # 计算位移的模长
        normalized_vectors = displacement_vectors / norms  # 归一化为单位向量
        unit_vectors.append(normalized_vectors)  # 保存单位向量

    # 将所有时间帧的单位向量拼接起来
    unit_vectors = np.concatenate(unit_vectors, axis=0)  # 合并为一个大数组
    avg_unit_vector = np.mean(unit_vectors, axis=0)  # 计算所有单位向量的平均
    avg_unit_vector /= np.linalg.norm(avg_unit_vector)  # 再次归一化，确保是单位向量

    # 计算每帧的位移标量
    scales = []
    for t in range(T):
        displacement = np.mean(tracks_3d[t] - tracks_3d[0], axis=0)  # 当前帧和初始帧的平均位移
        scale_t = np.dot(displacement, avg_unit_vector)  # 投影到单位向量上的标量
        scales.append(scale_t)
        
    # 进一步的进行一轮筛选
    if False:
        keep_ratio = 0.6
        sorted_indices = np.argsort(np.abs(scales))
        num_keep = int(T * keep_ratio)  # 保留的帧数
        start_idx = (T - num_keep) // 2  # 起始索引
        valid_indices = sorted_indices[start_idx : start_idx + num_keep]  # 中间部分的索引
        
        filtered_tracks = tracks_3d[valid_indices]  # 过滤掉两端数据的帧
        filtered_unit_vectors = []
        for t in range(1, len(filtered_tracks)):
            displacement_vectors = filtered_tracks[t] - tracks_3d[0]  # 当前帧和初始帧的位移
            norms = np.linalg.norm(displacement_vectors, axis=1, keepdims=True)  # 计算位移的模长
            normalized_vectors = displacement_vectors / norms  # 归一化为单位向量
            filtered_unit_vectors.append(normalized_vectors)  # 保存单位向量

        filtered_unit_vectors = np.concatenate(filtered_unit_vectors, axis=0)  # 合并为一个大数组
        filtered_avg_unit_vector = np.mean(filtered_unit_vectors, axis=0)  # 重新计算单位向量
        filtered_avg_unit_vector /= np.linalg.norm(filtered_avg_unit_vector)  # 再次归一化

        filtered_scales = []
        for t in range(T):
            displacement = np.mean(tracks_3d[t] - tracks_3d[0], axis=0)  # 当前帧和初始帧的平均位移
            scale_t = np.dot(displacement, filtered_avg_unit_vector)  # 投影到单位向量上的标量
            filtered_scales.append(scale_t)

        return avg_unit_vector, filtered_avg_unit_vector, np.array(scales), np.array(filtered_scales)
    return avg_unit_vector, np.array(scales)

def estimate_R_w_corr(trajectory):
    """
    通过轨迹估计旋转轴和旋转角度
    :param trajectory: 形状为(T, M, 3)的numpy数组，T是时间步数，M是点的数量
    :return: 旋转轴单位向量 (3,) 和旋转角度 (rad)
    """
    # 假设轨迹的起始和结束位置为旋转后的点云位置
    initial_positions = trajectory[0]  # 第一个时间步的点
    final_positions = trajectory[-1]  # 最后一个时间步的点
    
    # 计算初始位置和最终位置之间的旋转
    H = np.dot(initial_positions.T, final_positions)  # 计算协方差矩阵
    U, _, Vt = svd(H)  # 奇异值分解
    R = np.dot(U, Vt)  # 旋转矩阵
    
    # 从旋转矩阵计算旋转轴和角度
    angle = np.arccos((np.trace(R) - 1) / 2)  # 旋转角度
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # 旋转轴
    axis /= np.linalg.norm(axis)  # 归一化旋转轴
    
    return axis, angle