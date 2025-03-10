"""
Input:
    T, M, 3 的点云轨迹
    
Output:
    输出平移关节参数和误差统计
    输出旋转关节参数和误差统计

"""
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from embodied_analogy.visualization.vis_tracks_3d import (
    vis_tracks3d_napari,
    vis_pointcloud_series_napari
)
from embodied_analogy.utility.utils import napari_time_series_transform

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
    # reconstructed_tracks = np.expand_dims(tracks_3d[0], axis=0) + np.outer(scales, avg_unit_vector).reshape(T, 1, 3) # T, M, 3
    # est_loss = np.mean(np.linalg.norm(reconstructed_tracks - tracks_3d, axis=2))  # 计算点对点 L2 误差的平均值
    
        # 初始化每帧的位移标量
    scales = np.array(scales)
    scales_init = torch.from_numpy(scales).cuda().requires_grad_()
    avg_unit_vector_init = torch.from_numpy(avg_unit_vector).cuda().requires_grad_()

    # 定义损失函数
    def loss_function_torch(scales, avg_unit_vector):
        avg_unit_vector = avg_unit_vector / torch.norm(avg_unit_vector)
        reconstructed_tracks = torch.from_numpy(tracks_3d[0][None]).cuda() + (scales[:, None] * avg_unit_vector).unsqueeze(1)  # T, M, 3
        est_loss = torch.mean(torch.norm(reconstructed_tracks - torch.from_numpy(tracks_3d).cuda(), dim=2))  # 计算点对点 L2 误差的平均值
        return est_loss

    # 设置优化器
    optimizer = torch.optim.Adam([scales_init, avg_unit_vector_init], lr=1e-2) # 1 cm

    # 运行优化
    num_iterations = 100
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function_torch(scales_init, avg_unit_vector_init)
        print("coarse loss: ", loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        est_loss = loss_function_torch(scales_init, avg_unit_vector_init)

    scales = scales_init.detach().cpu().numpy()
    avg_unit_vector_init = avg_unit_vector_init / torch.norm(avg_unit_vector_init)
    avg_unit_vector = avg_unit_vector_init.detach().cpu().numpy()
    
    if visualize:
        reconstructed_tracks = np.expand_dims(tracks_3d[0], axis=0) + np.outer(scales, avg_unit_vector).reshape(T, 1, 3) # T, M, 3
        
        import napari
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "coarse translation estimation"
        
        viewer.add_points(napari_time_series_transform(tracks_3d), size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")
        viewer.add_points(napari_time_series_transform(reconstructed_tracks), size=0.01, name='renconstructed tracks 3d', opacity=0.8, face_color="red")
        
        napari.run()
        
    return avg_unit_vector, scales, est_loss

def coarse_R_from_tracks_3d(tracks_3d, visualize=False):
    """
    通过所有时间帧的点轨迹估计旋转轴，并通过优化方法求解每帧的旋转角度
    :param tracks_3d: 形状为 (T, M, 3) 的 numpy 数组, T 是时间步数, M 是点的数量
    :return: 旋转轴的单位向量 (3,), 每帧的旋转角度数组 (T,), 以及估计误差 est_loss
    """
    # TODO: 改得更简单一些
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    T, M, _ = tracks_3d.shape
    relative_rotations = []  # 存储所有相对于初始帧的旋转矩阵
    
    # 计算每一帧相对于初始帧的旋转矩阵
    for t in range(1, T):
        U, _, Vt = np.linalg.svd(tracks_3d[t].T @ tracks_3d[0])
        R_t = U @ Vt  # 计算旋转矩阵
        if np.linalg.det(R_t) < 0:  # 保证旋转矩阵的正定性
            U[:, -1] *= -1
            R_t = U @ Vt
        relative_rotations.append(R_t)
    
    # 计算所有旋转矩阵的平均旋转轴
    rotation_axes = []
    for R_t in relative_rotations:
        r = R.from_matrix(R_t)
        axis_angle = r.as_rotvec()  # 旋转向量
        axis = axis_angle / np.linalg.norm(axis_angle)  # 归一化旋转轴
        rotation_axes.append(axis)
    
    unit_vector_axis = np.mean(rotation_axes, axis=0)  # 计算所有旋转轴的平均
    unit_vector_axis /= np.linalg.norm(unit_vector_axis)  # 归一化旋转轴
    
    # 得到 angles 的初始值, 然后通过优化的方法更新
    angles_init = [0.0]  # 初始帧角度为0
    for R_t in relative_rotations:
        projected_rotation_vector = R.from_matrix(R_t).as_rotvec()
        angle = np.dot(projected_rotation_vector, unit_vector_axis)  # 计算在估计旋转轴上的旋转量
        angles_init.append(angle)
    angles_init = np.array(angles_init)
    
    def loss_function_torch(unit_vector_axis, angles):
        unit_vector_axis = unit_vector_axis / torch.norm(unit_vector_axis)
        est_loss = 0
        for t in range(T):
            theta = angles[t]
            skew_v = torch.tensor([[0, -unit_vector_axis[2], unit_vector_axis[1]],
                                    [unit_vector_axis[2], 0, -unit_vector_axis[0]],
                                    [-unit_vector_axis[1], unit_vector_axis[0], 0]], device="cuda")
            R_reconstructed = torch.eye(3, device="cuda") + torch.sin(theta) * skew_v + (1 - torch.cos(theta)) * (skew_v @ skew_v)
            reconstructed_track = (R_reconstructed @ torch.from_numpy(tracks_3d[0].T).float().cuda()).T
            est_loss += torch.mean(torch.norm(reconstructed_track - torch.from_numpy(tracks_3d[t]).cuda(), dim=1))
        return est_loss / T
    
    # 初始化 angles 并设置优化器
    angles = torch.from_numpy(angles_init).float().cuda().requires_grad_()
    unit_vector_axis = torch.from_numpy(unit_vector_axis).float().cuda().requires_grad_()
    optimizer = torch.optim.Adam([angles, unit_vector_axis], lr=1e-3)
    
    # 运行优化
    num_iterations = 100
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function_torch(unit_vector_axis, angles)
        print("coarse loss: ", loss.item())
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        est_loss = loss_function_torch(unit_vector_axis, angles)
    
    angles = angles.detach().cpu().numpy()
    unit_vector_axis = unit_vector_axis.detach().cpu().numpy()
    
    if visualize:
        # 绿色代表 moving part, 红色代表 reconstructed moving part
        reconstructed_tracks = [(R.from_rotvec(angles[t] * unit_vector_axis).as_matrix() @ tracks_3d[0].T).T for t in range(T)]
        
        import napari
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "coarse R estimation"
        
        viewer.add_points(napari_time_series_transform(tracks_3d), size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")
        viewer.add_points(napari_time_series_transform(reconstructed_tracks), size=0.01, name='renconstructed tracks 3d', opacity=0.8, face_color="red")
        
        napari.run()
    
    return unit_vector_axis, angles, est_loss
    
    
def coarse_joint_estimation(tracks_3d, visualize=False):
    """
    tracks_3d: (T, M, 3)
    """
    t_axis, t_states, t_est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize)
    # R_axis, R_states, R_est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize)
    R_axis, R_states, R_est_loss = None, None, 1e6
    
    print(f"t_est_loss: {t_est_loss}, R_est_loss: {R_est_loss}")
    # TODO：记得改回来
    if t_est_loss < R_est_loss:
        joint_type = "prismatic"
        joint_axis = t_axis
        joint_states = t_states
    else:
        joint_type = "revolute"
        joint_axis = R_axis
        joint_states = R_states
        
    return joint_type, joint_axis, joint_states


def generate_rotated_points(base_points, axis, angles, noise_std=0.01):
    """
    通过给定旋转轴和角度生成旋转后的点云，并加入噪声
    :param base_points: 初始点云 (M, 3)
    :param axis: 旋转轴 (3,)
    :param angles: 每一帧的旋转角度 (T,)
    :param noise_std: 每个点的噪声标准差
    :return: 每一帧的点云 (T, M, 3)
    """
    T = len(angles)
    rotated_points = []
    
    for t in range(T):
        # 生成旋转矩阵
        r = R.from_rotvec(angles[t] * axis)
        rotation_matrix = r.as_matrix()
        
        # 旋转点云
        rotated = base_points @ rotation_matrix.T
        
        # 添加噪声
        noise = np.random.normal(0, noise_std, rotated.shape)
        rotated_points.append(rotated + noise)
    
    return np.array(rotated_points)

def test_coarse_R_from_tracks_3d():
    """
    测试 coarse_R_from_tracks_3d 函数
    """
    # 设置参数
    T = 10  # 10 个时间步
    M = 50   # 5 个点
    true_axis = np.array([0, 0, 1])  # 真正的旋转轴是 Z 轴
    true_angles = np.linspace(0, np.pi / 4, T)  # 旋转角度从 0 到 pi/4
    noise_std = 0.05  # 加噪声的标准差

    # 初始化第0帧点云
    base_points = np.random.rand(M, 3)  # 随机生成 5 个点的 3D 坐标
    base_points[:, 0] = 0
    
    # 生成带噪声的点云数据
    tracks_3d = generate_rotated_points(base_points, true_axis, true_angles, noise_std)
    
    # 调用 coarse_R_from_tracks_3d 函数
    unit_vector_axis, angles, est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize=True)
    
    # 打印优化后的旋转轴和角度
    print("优化后的旋转轴:", unit_vector_axis)
    print("优化后的角度:", angles)
    
    # 计算与真实旋转轴的夹角
    axis_error = np.arccos(np.dot(unit_vector_axis, true_axis))  # 计算两旋转轴之间的夹角（弧度）
    print(f"旋转轴误差: {np.degrees(axis_error)}°")
    
    # 计算角度误差
    angle_error = np.abs(angles - true_angles)  # 每一帧的角度误差
    print(f"每帧角度误差: {np.degrees(angle_error)}°")
    
    # 打印估计误差
    print(f"估计误差 (loss): {est_loss}")

def generate_translated_points(base_points, direction, scales, noise_std=0.01):
    """
    通过给定平移方向和标量生成平移后的点云，并加入噪声
    :param base_points: 初始点云 (M, 3)
    :param direction: 平移方向 (3,)
    :param scales: 每一帧的平移标量 (T,)
    :param noise_std: 每个点的噪声标准差
    :return: 每一帧的点云 (T, M, 3)
    """
    T = len(scales)
    translated_points = []
    
    for t in range(T):
        # 计算平移
        translation = scales[t] * direction
        translated = base_points + translation
        
        # 添加噪声
        noise = np.random.normal(0, noise_std, translated.shape)
        translated_points.append(translated + noise)
    
    return np.array(translated_points)

def test_coarse_t_from_tracks_3d():
    """
    测试 coarse_t_from_tracks_3d 函数
    """
    # 设置参数
    T = 10  # 10 个时间步
    M = 50   # 5 个点
    true_direction = np.array([1, 0, 0])  # 真正的平移方向是 X 轴
    true_scales = np.linspace(0, 1, T)  # 平移标量从 0 到 1
    noise_std = 0.1  # 加噪声的标准差

    # 初始化第0帧点云
    base_points = np.random.rand(M, 3)  # 随机生成 5 个点的 3D 坐标
    
    # 生成带噪声的点云数据
    tracks_3d = generate_translated_points(base_points, true_direction, true_scales, noise_std)
    
    # 调用 coarse_t_from_tracks_3d 函数
    avg_unit_vector, scales, est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize=True)
    
    # 打印优化后的平移方向和标量
    print("优化后的平移方向:", avg_unit_vector)
    print("优化后的平移标量:", scales)
    
    # 计算与真实平移方向的夹角
    direction_error = np.arccos(np.dot(avg_unit_vector, true_direction))  # 计算两平移方向之间的夹角（弧度）
    print(f"平移方向误差: {np.degrees(direction_error)}°")
    
    # 计算标量误差
    scale_error = np.abs(scales - true_scales)  # 每一帧的标量误差
    print(f"每帧标量误差: {scale_error}")
    
    # 打印估计误差
    print(f"估计误差 (loss): {est_loss}")


if __name__ == "__main__":
    test_coarse_R_from_tracks_3d()
    # 调用测试函数
    # test_coarse_t_from_tracks_3d()