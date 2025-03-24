import napari
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_analogy.utility.utils import napari_time_series_transform, fit_plane_normal, remove_dir_component
from embodied_analogy.estimation.scheduler import Scheduler

def coarse_t_from_tracks_3d(tracks_3d, visualize=False):
    """
    通过所有时间帧的位移变化估计平移方向，并计算每帧沿该方向的位移标量
    :param tracks_3d: 形状为(T, M, 3)的numpy数组, T是时间步数, M是点的数量
    :return: 平移方向的平均单位向量 (3,), 每帧的位移标量数组 (T,)
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    tracks_3d = np.copy(tracks_3d)
    T, M, _ = tracks_3d.shape
    
    joint_dir = None
    joint_states = None
    # 3, 默认用 frame 0 的平均位置作为关节轴的起始位置
    joint_start = tracks_3d[0].mean(axis=0) 

    # 遍历所有帧（从t2到tN，与t1的差异）
    centers = tracks_3d.mean(axis=1) # T, 3
    centers = centers - centers[0] #, 3
    
    # 用最简单的方法进行初始估计
    joint_dir = centers[-1] - centers[0]
    joint_dir = joint_dir / np.linalg.norm(joint_dir)
    joint_states = np.array([np.dot(centers[i] - centers[0], joint_dir) for i in range(T)])

    # 初始化每帧的位移标量
    joint_dir = torch.from_numpy(joint_dir).cuda().requires_grad_()
    joint_states = torch.from_numpy(joint_states[1:]).cuda().requires_grad_() # T-1

    # 定义损失函数
    def loss_function_torch(joint_states, joint_dir):
        joint_dir = joint_dir / torch.norm(joint_dir)
        base_frame = torch.from_numpy(tracks_3d[0][None]).cuda()
        reconstructed_tracks = base_frame + (joint_states[:, None] * joint_dir).unsqueeze(1)  # T-1, M, 3
        est_loss = torch.mean(torch.norm(reconstructed_tracks - torch.from_numpy(tracks_3d[1:]).cuda(), dim=2))  # 计算点对点 L2 误差的平均值
        return est_loss

    # 设置优化器
    optimizer = torch.optim.Adam([joint_states, joint_dir], lr=1e-2) # 1 cm
    scheduler = Scheduler(
        optimizer=optimizer,
        lr_update_factor=0.5,
        lr_scheduler_patience=3,
        early_stop_patience=10,
    )

    # 运行优化
    num_iterations = 300
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function_torch(joint_states, joint_dir)
        print(f"[{i}/{num_iterations}] coarse t loss: ", loss.item())
        
        joint_states_tmp = joint_states.detach().cpu().numpy()
        # 给最开始插入 0
        joint_states_tmp = np.insert(joint_states_tmp, 0, 0)
        joint_dir_tmp = joint_dir.detach().cpu().numpy()
        joint_dir_tmp = joint_dir_tmp / np.linalg.norm(joint_dir_tmp)
        cur_state_dict = {
            "joint_states": joint_states_tmp,
            "joint_start": joint_start,
            "joint_dir": joint_dir_tmp,
        }
        should_early_stop = scheduler.step(loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f"\t lr:", cur_lr)
        
        if should_early_stop:
            print("EARLY STOP")
            break
        
        loss.backward()
        optimizer.step()

    if visualize:
        joint_states = np.copy(scheduler.best_state_dict["joint_states"])
        joint_dir = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_start = np.copy(scheduler.best_state_dict["joint_start"])
        reconstructed_tracks = np.expand_dims(tracks_3d[0], axis=0) + np.outer(joint_states, joint_dir).reshape(T, 1, 3) # T, M, 3
        
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "coarse translation estimation"
        
        # 改变坐标系
        joint_start[-1] *= -1
        joint_dir[-1] *= -1
        tracks_3d[..., -1] *= -1
        reconstructed_tracks[..., -1] *= -1
        
        viewer.add_points(napari_time_series_transform(tracks_3d), size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")
        viewer.add_points(napari_time_series_transform(reconstructed_tracks), size=0.01, name='renconstructed tracks 3d', opacity=0.8, face_color="red")

        # 绘制一下 joint start 和 joint axis
        viewer.add_shapes(
            data=np.array([joint_start, joint_start + joint_dir * 0.2]),
            name="prismatic joint",
            shape_type="line",
            edge_width=0.005,
            face_color="blue",
            edge_color="blue"
        )
        viewer.add_points(
            data=joint_start,
            name="joint start",
            size=0.02,
            face_color="blue",
            border_color="red",
        )
        napari.run()
        
    return scheduler.best_state_dict, scheduler.best_loss


def coarse_R_from_tracks_3d(tracks_3d, visualize=False):
    # 使用一个 frame 的点云估计一整个 normal
    """
    通过所有时间帧的点轨迹估计旋转轴，并通过优化方法求解每帧的旋转角度
    :param tracks_3d: 形状为 (T, M, 3) 的 numpy 数组, T 是时间步数, M 是点的数量
    :return: 旋转轴的单位向量 (3,), 每帧的旋转角度数组 (T,), 以及估计误差 est_loss
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    tracks_3d = np.copy(tracks_3d)
    T, M, _ = tracks_3d.shape
    tracks_3d_mean = tracks_3d.mean(axis=1) # T, 1, 3
    
    # 注意 joint_dir 应该满足垂直于所有 tracks_3d_diff 的方向, 因此可以复用 fit_normal 函数来求解
    tracks_3d_diff = tracks_3d_mean[1:, ...] - tracks_3d_mean[:-1, ...] # (T-1), M, 3
    tracks_3d_flow = tracks_3d_diff.reshape(-1, 3) # N, 3
    _, joint_dir = fit_plane_normal(tracks_3d_flow)
    
    # 对于 joint_start 应该满足 (joint_start - tracks_3d_mid) 垂直于 tracks_3d_diff
    # 可以通过最小二乘对 joint_start 直接求解， 即 (joint_start - tracks_3d_mid) * tracks_3d_diff = 0
    # 即 tracks_3d_diff @ joint_start = (tracks_3d_mid * tracks_3d_diff).sum(axis=-1)
    tracks_3d_mid = (tracks_3d_mean[1:, ...] + tracks_3d_mean[:-1, ...]) / 2.0
    tracks_3d_mid = tracks_3d_mid.reshape(-1, 3) # N, 3
    joint_start, _, _, _ = np.linalg.lstsq(tracks_3d_flow, (tracks_3d_flow * tracks_3d_mid).sum(axis=-1), rcond=None)
    joint_start = remove_dir_component(joint_start, joint_dir)
    # print(joint_dir, joint_start)
    
    # 接着估计出每一帧对应的 angle, 这里可能需要进行一个方向的对齐
    joint_states = np.zeros(T)  # 初始化角度数组，第一帧角度为0
    for t in range(1, T):
        # 计算初始帧和当前帧的点云中心
        center_0 = tracks_3d[0].mean(axis=0)
        center_t = tracks_3d[t].mean(axis=0)
        
        # 减去 joint_start 向量
        diff_0 = center_0 - joint_start
        diff_t = center_t - joint_start
        
        # 并减去 joint_dir 分量
        diff_0 = remove_dir_component(diff_0, joint_dir)
        diff_t = remove_dir_component(diff_t, joint_dir)
        
        # 进行归一化
        diff_0 /= np.linalg.norm(diff_0)
        diff_t /= np.linalg.norm(diff_t)
        
        # diff_0 * diff_t = cos(angle)
        angle = np.arccos(np.clip(np.dot(diff_0, diff_t), -1.0, 1.0)) # [0, pi]
        joint_states[t] = angle
        
        if t == T-1:
            if np.dot(np.cross(diff_0, diff_t), joint_dir) < 0:
                joint_dir = -joint_dir
        
    # print("before torch optimization")
    # print("joint_dir: ", joint_dir)
    # print("joint_start: ", joint_start)
    # print("joint_states: ", joint_states)
    
    # 初始化 joint_states 和 joint_start，并设置优化器
    joint_states = torch.from_numpy(joint_states[1:]).float().cuda().requires_grad_() # T-1
    joint_dir = torch.from_numpy(joint_dir).float().cuda().requires_grad_()
    joint_start = torch.from_numpy(joint_start).float().cuda().requires_grad_()
    
    def loss_function_torch(joint_dir, joint_states, joint_start):
        # 归一化 joint_dir
        joint_dir = joint_dir / torch.norm(joint_dir)
        # 创建 skew_v 矩阵
        skew_v = torch.zeros((3, 3), device="cuda")
        skew_v[0, 1] = -joint_dir[2]
        skew_v[0, 2] = joint_dir[1]
        skew_v[1, 0] = joint_dir[2]
        skew_v[1, 2] = -joint_dir[0]
        skew_v[2, 0] = -joint_dir[1]
        skew_v[2, 1] = joint_dir[0]
        # 计算 R_reconstructed 矩阵
        theta = joint_states.unsqueeze(-1).unsqueeze(-1)  # 将 theta 扩展为 (T-1, 1, 1) 形状
        # (T-1, 3, 3)
        R_reconstructed = (torch.eye(3, device="cuda") + torch.sin(theta) * skew_v + (1 - torch.cos(theta)) * (skew_v @ skew_v))  
        # 计算 translated_initial
        translated_initial = torch.from_numpy(tracks_3d[0]).float().cuda() - joint_start  # N, 3
        # 计算重建轨迹
        reconstructed_track = torch.einsum('t a b, n b -> t n a', R_reconstructed, translated_initial) + joint_start
        # 计算损失
        est_loss = torch.mean(torch.norm(reconstructed_track - torch.from_numpy(tracks_3d[1:]).cuda(), dim=2))  # (T-1, N)
        return est_loss 

    optimizer = torch.optim.Adam([joint_states, joint_dir, joint_start], lr=1e-2) # 1 cm
    
    # lr = 1e-2
    # dir_lr = 1e-2
    # optimizer = torch.optim.Adam(
    #     params=[
    #         {'params': joint_states, 'lr': lr},
    #         {'params': joint_dir, 'lr': dir_lr},
    #         {'params': joint_start, 'lr': lr}
    #     ]
    # )
    scheduler = Scheduler(
        optimizer=optimizer,
        lr_update_factor=0.5,
        lr_scheduler_patience=3,
        early_stop_patience=10,
    )
    
    # 运行优化
    num_iterations = 300
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function_torch(joint_dir, joint_states, joint_start)
        print(f"[{i}/{num_iterations}] coarse R loss: ", loss.item())
        
        joint_states_tmp = joint_states.detach().cpu().numpy()
        joint_states_tmp = np.insert(joint_states_tmp, 0, 0)
        joint_dir_tmp = joint_dir.detach().cpu().numpy()
        joint_dir_tmp = joint_dir_tmp / np.linalg.norm(joint_dir_tmp)
        joint_start_tmp = joint_start.detach().cpu().numpy()
        cur_state_dict = {
            "joint_states": joint_states_tmp,
            "joint_dir": joint_dir_tmp,
            "joint_start": joint_start_tmp
        }
        should_early_stop = scheduler.step(loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f"\t lr:", cur_lr)
        
        if should_early_stop:
            print("EARLY STOP")
            break
        
        loss.backward()
        optimizer.step()
        
    # print("after torch optimization")
    # print("joint_dir: ", scheduler.best_state_dict["joint_dir"])
    # print("joint_start: ", scheduler.best_state_dict["joint_start"])
    # print("joint_states: ", scheduler.best_state_dict["joint_states"])
    
    if visualize:
        # NOTE: 由于 napari 的显示是左手坐标系, 因此需要把所有三维数据的 z 轴乘 -1
        joint_states = np.copy(scheduler.best_state_dict["joint_states"])
        joint_dir = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_start = np.copy(scheduler.best_state_dict["joint_start"])
        
        # 绿色代表 moving part, 红色代表 reconstructed moving part
        reconstructed_tracks = [(R.from_rotvec(joint_states[t] * joint_dir).as_matrix() @ (tracks_3d[0] - joint_start).T).T + joint_start for t in range(T)]
        reconstructed_tracks = np.array(reconstructed_tracks)
        
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "coarse R estimation"
        
        joint_dir[-1] = -joint_dir[-1]
        joint_start[-1] = -joint_start[-1]
        tracks_3d[..., -1] = -tracks_3d[..., -1]
        reconstructed_tracks[..., -1] = -reconstructed_tracks[..., -1]
        
        viewer.add_points(napari_time_series_transform(tracks_3d), size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")
        viewer.add_points(napari_time_series_transform(reconstructed_tracks), size=0.01, name='renconstructed tracks 3d', opacity=0.8, face_color="red")
        
        # 绘制一下 joint start 和 joint axis
        viewer.add_shapes(
            data=np.array([joint_start, joint_start + joint_dir * 0.2]),
            name="revolute joint",
            shape_type="line",
            edge_width=0.005,
            edge_color="blue",
            face_color="blue",
        )
        viewer.add_shapes(
            data=0.1 + np.array([np.array([0, 0, 0]), np.array([1, 0, 0]) * 0.2]),
            name="origin_x",
            shape_type="line",
            edge_width=0.005,
            edge_color="red",
            face_color="red",
        )
        viewer.add_shapes(
            data=0.1 + np.array([np.array([0, 0, 0]), np.array([0, 1, 0]) * 0.2]),
            name="origin_y",
            shape_type="line",
            edge_width=0.005,
            edge_color="green",
            face_color="green",
        )
        viewer.add_shapes(
            data=0.1 + np.array([np.array([0, 0, 0]), np.array([0, 0, -1]) * 0.2]),
            name="origin_z",
            shape_type="line",
            edge_width=0.005,
            edge_color="blue",
            face_color="blue",
        )
        viewer.add_points(
            data=joint_start,
            name="joint start",
            size=0.02,
            face_color="blue",
            border_color="red",
        )
        napari.run()
        
    return scheduler.best_state_dict, scheduler.best_loss


def coarse_estimation(tracks_3d, visualize=False):
    """
    根据 tracks3d 估计出初始的 joint 状态, 要求 joint_state 的初始值是0, 且随着轨迹增加
    (如果是旋转的话需要满足右手定则)
    tracks_3d: (T, M, 3)
    """
    t_state_dict, t_est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize)
    R_state_dict, R_est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize)
    
    print(f"t_est_loss: {t_est_loss}, R_est_loss: {R_est_loss}")
    
    if t_est_loss < R_est_loss:
        print("select as prismatic joint")
        t_state_dict["joint_type"] = "prismatic"
        return t_state_dict
    else:
        print("select as revolute joint")
        R_state_dict["joint_type"] = "revolute"
        return R_state_dict


def generate_rotated_points(base_points, axis, angles, joint_start_point, noise_std=0.01):
    """
    通过给定旋转轴和角度生成旋转后的点云，并加入噪声
    :param base_points: 初始点云 (M, 3)
    :param axis: 旋转轴 (3,)
    :param angles: 每一帧的旋转角度 (T,)
    :param joint_start_point: 旋转轴的起始点 (3,)
    :param noise_std: 每个点的噪声标准差
    :return: 每一帧的点云 (T, M, 3)
    """
    T = len(angles)
    rotated_points = []
    
    # 将旋转轴起始点移至原点
    translated_points = base_points - joint_start_point
    
    for t in range(T):
        # 生成旋转矩阵
        r = R.from_rotvec(angles[t] * axis)
        rotation_matrix = r.as_matrix()
        
        # 旋转点云
        rotated = translated_points @ rotation_matrix.T
        
        # 添加噪声
        noise = np.random.normal(0, noise_std, rotated.shape)
        rotated_points.append(rotated + noise)
    
    # 将旋转后的点云移回旋转轴起始点
    return np.array(rotated_points) + joint_start_point


def test_coarse_R_from_tracks_3d():
    """
    测试 coarse_R_from_tracks_3d 函数
    """
    # 设置参数
    T = 10  # 10 个时间步
    M = 1000   # 5 个点
    true_axis = np.array([0.6, 0.8, 0.])  # 真正的旋转轴是 Z 轴
    true_angles = np.linspace(0, np.pi / 3, T)  # 旋转角度从 0 到 pi/4
    noise_std = 0.00  # 加噪声的标准差
    joint_start_point = np.array([1, 1, 1])  # 旋转轴的起始点
    joint_start_point = remove_dir_component(joint_start_point, true_axis)
    print("gt axis: ", true_axis)
    print("gt_start: ", joint_start_point)
    print("gt states: ", true_angles)

    # 初始化第0帧点云
    base_points = np.random.rand(M, 3)  # 随机生成 5 个点的 3D 坐标
    base_points[:, 2] = base_points[:, 0] ** 2 + base_points[:, 1] ** 2
    
    # 生成带噪声的点云数据
    tracks_3d = generate_rotated_points(base_points, true_axis, true_angles, joint_start_point, noise_std)
    
    # 调用 coarse_R_from_tracks_3d 函数
    best_state, est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize=True)
    unit_vector_axis, angles = best_state["joint_dir"], best_state["joint_states"]
    
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
    noise_std = 0.01  # 加噪声的标准差

    # 初始化第0帧点云
    base_points = np.random.rand(M, 3)  # 随机生成 5 个点的 3D 坐标
    
    # 生成带噪声的点云数据
    tracks_3d = generate_translated_points(base_points, true_direction, true_scales, noise_std)
    
    # 调用 coarse_t_from_tracks_3d 函数
    best_state, est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize=True)
    joint_dir, scales = best_state["joint_dir"], best_state["joint_states"]
    
    # 打印优化后的平移方向和标量
    print("优化后的平移方向:", joint_dir)
    print("优化后的平移标量:", scales)
    
    # 计算与真实平移方向的夹角
    direction_error = np.arccos(np.dot(joint_dir, true_direction))  # 计算两平移方向之间的夹角（弧度）
    print(f"平移方向误差: {np.degrees(direction_error)}°")
    
    # 计算标量误差
    scale_error = np.abs(scales - true_scales)  # 每一帧的标量误差
    print(f"每帧标量误差: {scale_error}")
    
    # 打印估计误差
    print(f"估计误差 (loss): {est_loss}")


def test_coarse_R():
    # tracks_3d = np.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/reconstruct/rotate_track_open.npy")
    # tracks_3d = tracks_3d[::-1, ...]
    tracks_3d = np.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/reconstruct/rotate_track_close.npy")
    # 调用 coarse_R_from_tracks_3d 函数
    best_state, est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize=True)
    
    
if __name__ == "__main__":
    # test_coarse_R_from_tracks_3d()
    test_coarse_R()
    # test_coarse_t_from_tracks_3d()
    