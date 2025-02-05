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
from scipy.spatial.transform import Rotation as R
from embodied_analogy.visualization.vis_tracks_3d import (
    vis_tracks3d_napari,
    vis_pointcloud_series_napari
)

def coarse_t_from_tracks_3d(tracks_3d, visualize=False):
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

    # 计算重投影误差 loss
    reconstructed_tracks = np.expand_dims(tracks_3d[0], axis=0) + np.outer(scales, avg_unit_vector).reshape(T, 1, 3) # T, M, 3
    est_loss = np.mean(np.linalg.norm(reconstructed_tracks - tracks_3d, axis=2))  # 计算点对点 L2 误差的平均值
    
    if visualize:
        # 绿色代表 moving part, 红色代表 renconstructed moving part
        colors = np.vstack((np.tile([0, 1, 0], (M, 1)), np.tile([1, 0, 0], (M, 1)))) # 2M, 3
        vis_tracks3d_napari(np.concatenate([tracks_3d, reconstructed_tracks], axis=1), colors)
    return avg_unit_vector, np.array(scales), est_loss

def coarse_R_from_tracks_3d(tracks_3d, visualize=False):
    """
    通过所有时间帧的点轨迹估计旋转轴，并计算每帧的旋转角度
    :param tracks_3d: 形状为 (T, M, 3) 的 numpy 数组, T 是时间步数, M 是点的数量
    :return: 旋转轴的单位向量 (3,), 每帧的旋转角度数组 (T,), 以及估计误差 est_loss
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    T, M, _ = tracks_3d.shape
    relative_rotations = []  # 存储所有相对于初始帧的旋转矩阵
    
    # 计算每一帧相对于初始帧的旋转矩阵
    for t in range(1, T):
        # 由于这是纯旋转变换, 所以不需要减去中心值
        U, _, Vt = np.linalg.svd(tracks_3d[t].T @ tracks_3d[0])
        R_t = U @ Vt  # 计算旋转矩阵
        if np.linalg.det(R_t) < 0:  # 保证旋转矩阵的正定性
            U[:, -1] *= -1
            R_t = U @ Vt
        relative_rotations.append(R_t)
    
    # 计算所有旋转矩阵的平均旋转轴
    rotation_axes = []
    angles = []
    for R_t in relative_rotations:
        r = R.from_matrix(R_t)
        axis_angle = r.as_rotvec()  # 旋转向量
        axis = axis_angle / np.linalg.norm(axis_angle)  # 归一化旋转轴
        angle = np.linalg.norm(axis_angle)  # 旋转角度
        rotation_axes.append(axis)
        angles.append(angle)
    
    unit_vector_axis = np.mean(rotation_axes, axis=0)  # 计算所有旋转轴的平均
    unit_vector_axis /= np.linalg.norm(unit_vector_axis)  # 归一化旋转轴
    
    # 重新计算每帧旋转角度
    angles = [0.0]  # 初始帧角度为0
    for R_t in relative_rotations:
        projected_rotation_vector = R.from_matrix(R_t).as_rotvec()
        angle = np.dot(projected_rotation_vector, unit_vector_axis)  # 计算在估计旋转轴上的旋转量
        angles.append(angle)
    
    angles = np.array(angles)  # 转换为数组
    
    # 计算重投影误差 est_loss
    est_loss = 0
    reconstructed_tracks = []
    for t in range(T):
        R_reconstructed = R.from_rotvec(angles[t] * unit_vector_axis).as_matrix()
        reconstructed_track = (R_reconstructed @ tracks_3d[0].T).T # T, M, 3
        est_loss += np.mean(np.linalg.norm(reconstructed_track - tracks_3d[t], axis=1))
        reconstructed_tracks.append(reconstructed_track)
    est_loss /= T  # 计算所有帧的平均误差
    
    if visualize:
        # 绿色代表 moving part, 红色代表 renconstructed moving part
        # 绿色代表 moving part, 红色代表 renconstructed moving part
        colors = np.vstack((np.tile([0, 1, 0], (M, 1)), np.tile([1, 0, 0], (M, 1)))) # 2M, 3
        vis_tracks3d_napari(np.concatenate([tracks_3d, np.array(reconstructed_tracks)], axis=1), colors)
    return unit_vector_axis, angles, est_loss