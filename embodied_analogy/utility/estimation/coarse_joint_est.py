import torch
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_analogy.utility.utils import (
    napari_time_series_transform,
    fit_plane_normal,
    remove_dir_component,
    initialize_napari
)
initialize_napari()
from embodied_analogy.utility.estimation.scheduler import Scheduler


def random_rotation_matrix(angle_range=60):
    """
    angle_range: 旋转的角度值大小, in degree
    
    生成一个随机的旋转矩阵    
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, angle_range)
    angle = np.deg2rad(angle)
    
    rotation_vector = angle * axis
    rotation = R.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


def sample_around_given_joint_param(joint_start, joint_dir, num_sample=100):
    """
    在 joint_dict 附近进行一些采样, 返回一系列新的 joint_dict
    
    对于 pivot point 来说, 在该点附近进行采样
    对于 joint_dir 来说, 
    """
    assert num_sample >= 1
    np.random.seed(666)
    joint_starts, joint_dirs = [joint_start], [joint_dir]
    
    for _ in range(num_sample - 1):
        tmp_joint_start = joint_start + np.random.uniform(low=-0.5, high=0.5, size=(3, ))
        tmp_joint_dir = random_rotation_matrix(angle_range=60) @ joint_dir
        joint_starts.append(tmp_joint_start)
        joint_dirs.append(tmp_joint_dir)
    
    joint_starts, joint_dirs = np.array(joint_starts), np.array(joint_dirs)
    return joint_starts, joint_dirs


def get_init_joint_param(tracks_3d, joint_type):
    """
    tracks_3d: np.array([T, M, 3])
    
    根据 tracks_3d 先猜一个 joint_start 和 joint_dir 出来
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
        
    if joint_type == "prismatic":
        centers = tracks_3d.mean(axis=1) # T, 3
        joint_dir = centers[-1] - centers[0]
        joint_dir = joint_dir / np.linalg.norm(joint_dir)
        joint_start = tracks_3d[0].mean(axis=0) 
        
    elif joint_type == "revolute":
        tracks_3d_diff = tracks_3d[1:, ...] - tracks_3d[:-1, ...] # (T-1), M, 3
        tracks_3d_flow = tracks_3d_diff.reshape(-1, 3) # (T-1) * M, 3
        _, joint_dir = fit_plane_normal(tracks_3d_flow)
        
        min_idx = np.argmin(np.linalg.norm(tracks_3d_diff, axis=-1).mean(0))
        joint_start = tracks_3d[0, min_idx]
        
        joint_dir = adjust_joint_dir(
            tracks_3d=tracks_3d,
            joint_start=joint_start,
            joint_dir=joint_dir
        )
    
    return joint_start, joint_dir


def adjust_joint_dir(tracks_3d, joint_start, joint_dir):
    """
    因为我们默认 tracks_3d 是 open joint 的序列, 所以需要根据实际判断 joint_dir 是否需要反转
    当然仅仅需要对于 revolute 关节进行这个操作
    """
    # 在这里对 joint_dir 进行一个修正
    center_0 = tracks_3d[0].mean(axis=0)
    center_t = tracks_3d[-1].mean(axis=0)
    
    diff_0 = center_0 - joint_start
    diff_t = center_t - joint_start
    
    diff_0 /= np.linalg.norm(diff_0)
    diff_t /= np.linalg.norm(diff_t)
    
    if np.dot(np.cross(diff_0, diff_t), joint_dir) < 0:
        joint_dir = -joint_dir
    return joint_dir

    
def estimate_joint_state_given_joint_param(joint_start, joint_dir, joint_type, tracks_3d):
    """
    joint_start
    joint_dir
    joint_type: prismatic or revolute
    tracks_3d: np.array([T, M, 3])
    
    估计的策略是: 将每一帧抽象为一个点, 根据这个点计算 joint_state
    NOTE 这个函数并不要求 revolute 物品的 joint_dir 的方向正确
    return joint_dict, but with estimated joint_states
    """
    T, M, _ = tracks_3d.shape
    centers = tracks_3d.mean(axis = 1) # T, 3
    
    if joint_type == "prismatic":
        joint_states = np.array([np.dot(centers[i] - centers[0], joint_dir) for i in range(T)])
    else:
        joint_states = np.zeros(T)
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
            
            # NOTE diff_0 * diff_t = cos(angle)
            angle = np.arccos(np.clip(np.dot(diff_0, diff_t), -1.0, 1.0)) # [0, pi]
            joint_states[t] = angle
    return joint_states


def loss_function_torch(tracks_3d, joint_states, joint_dir, joint_type, joint_start):
    """
    tracks_3d: np.array([T, M, 3])
    joint_states: torch.Tensor([T-1], requires_grad=True)
    joint_dir: torch.Tensor([3], requires_grad=True)
    joint_start: torch.Tensor([3], requires_grad=True)
    joint_type: "prismatic" or "revolute"
    """
    if joint_type == "prismatic":
        joint_dir = joint_dir / torch.norm(joint_dir)
        base_frame = torch.from_numpy(tracks_3d[0][None]).cuda()
        reconstructed_tracks = base_frame + (joint_states[:, None] * joint_dir).unsqueeze(1)  # T-1, M, 3
        est_loss = torch.mean(torch.norm(reconstructed_tracks - torch.from_numpy(tracks_3d[1:]).cuda(), dim=2))  # 计算点对点 L2 误差的平均值
        return est_loss
    elif joint_type == "revolute":
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


def loss_function_torch_batch(tracks_3d, joint_states, joint_dir, joint_type, joint_start):
    """
    Batch version supporting [B, ...] dimensions
    
    tracks_3d: np.array([T, M, 3])
    joint_states: torch.Tensor([B, T-1], requires_grad=True)
    joint_dir: torch.Tensor([B, 3], requires_grad=True)
    joint_start: torch.Tensor([B, 3], requires_grad=True)
    joint_type: "prismatic" or "revolute"
    """
    device = "cuda"
    tracks_3d_tensor = torch.from_numpy(tracks_3d).float().cuda() # T, M, 3
    B = joint_states.shape[0]  # Batch size
    tracks_3d_tensor = tracks_3d_tensor.unsqueeze(0).expand(B, -1, -1, -1)
    
    if joint_type == "prismatic":
        # 归一化每个batch的joint_dir [B, 3]
        joint_dir = joint_dir / torch.norm(joint_dir, dim=1, keepdim=True)
        
        # 获取初始帧 [B, 1, M, 3]
        base_frame = tracks_3d_tensor[:, 0:1]
        
        # 重构轨迹: base_frame + joint_states * joint_dir
        # joint_states: [B, T-1] -> [B, T-1, 1, 1]
        # joint_dir: [B, 3] -> [B, 1, 1, 3]
        displacement = joint_states[:, :, None, None] * joint_dir[:, None, None, :] #  [B, T-1, 1, 3]
        reconstructed_tracks = base_frame + displacement #  [B, T-1, M, 3]
        
        # 计算与后续帧的L2误差 [B, T-1, M]
        errors = torch.norm(reconstructed_tracks - tracks_3d_tensor[:, 1:], dim=-1)
        
        # for optimization
        loss = torch.mean(errors)
        
        # for best selection 
        batchwise_loss = torch.mean(errors, dim=(1, 2)) # B
        best_loss, best_idx = torch.min(batchwise_loss, dim=0)
        return loss, best_loss, best_idx
        
    elif joint_type == "revolute":
        # 归一化每个batch的joint_dir [B, 3]
        joint_dir = joint_dir / torch.norm(joint_dir, dim=1, keepdim=True)
        
        # 创建批量skew-symmetric矩阵 [B, 3, 3]
        skew_v = torch.zeros((B, 3, 3), device=device)
        
        # 填充skew-symmetric矩阵
        skew_v[:, 0, 1] = -joint_dir[:, 2]
        skew_v[:, 0, 2] = joint_dir[:, 1]
        skew_v[:, 1, 0] = joint_dir[:, 2]
        skew_v[:, 1, 2] = -joint_dir[:, 0]
        skew_v[:, 2, 0] = -joint_dir[:, 1]
        skew_v[:, 2, 1] = joint_dir[:, 0]
        
        # 计算旋转矩阵 [B, T-1, 3, 3]
        theta = joint_states.unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1]
        eye_batch = torch.eye(3, device=device).repeat(B, 1, 1).unsqueeze(1)  # [B, 1, 3, 3]
        skew_v_batch = skew_v.unsqueeze(1)  # [B, 1, 3, 3]
        
        # 使用Rodrigues公式计算旋转矩阵
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        R_reconstructed = (
            eye_batch + 
            sin_theta * skew_v_batch + 
            (1 - cos_theta) * torch.matmul(skew_v_batch, skew_v_batch)
        ) # B, T-1, 3, 3
        
        # 计算平移后的初始位置 [B, M, 3]
        initial_points = tracks_3d_tensor[:, 0]  # [B, M, 3]
        translated_initial = initial_points - joint_start.unsqueeze(1)  # [B, M, 3]
        
        # 批量旋转点云 [B, T-1, M, 3] = [B, T-1, 3, 3] @ [B, M, 3]
        # einsum解释: b:batch, t:time, i,j:matrix dim, m:point index
        reconstructed_track = torch.einsum('b t i j, b m j -> b t m i', 
                                          R_reconstructed, 
                                          translated_initial)
        # 加回关节起点 [B, T-1, M, 3]
        reconstructed_track += joint_start.unsqueeze(1).unsqueeze(1)
        
        # 计算损失 [B, T-1, M]
        errors = torch.norm(reconstructed_track - tracks_3d_tensor[:, 1:], dim=-1)
        # for optimization
        loss = torch.mean(errors)
        
        # for best selection 
        batchwise_loss = torch.mean(errors, dim=(1, 2)) # B
        best_loss, best_idx = torch.min(batchwise_loss, dim=0)
        return loss, best_loss, best_idx
    
    
def coarse_t_from_tracks_3d(tracks_3d, visualize=False, logger=None):
    """
    通过所有时间帧的位移变化估计平移方向，并计算每帧沿该方向的位移标量
    :param tracks_3d: 形状为(T, M, 3)的numpy数组, T是时间步数, M是点的数量
    :return: 平移方向的平均单位向量 (3,), 每帧的位移标量数组 (T,)
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    joint_start, joint_dir = get_init_joint_param(
        tracks_3d=tracks_3d,
        joint_type="prismatic"
    )
        
    joint_states = estimate_joint_state_given_joint_param(
        joint_start=joint_start,
        joint_dir=joint_dir,
        joint_type="prismatic",
        tracks_3d=tracks_3d
    )

    # 将初始的 numpy 估计改为有梯度的参数
    joint_dir = torch.from_numpy(joint_dir).cuda().requires_grad_()
    joint_states = torch.from_numpy(joint_states[1:]).cuda().requires_grad_() # T-1
    joint_start = torch.from_numpy(joint_start).float().cuda().requires_grad_()

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
        loss = loss_function_torch(
            tracks_3d=tracks_3d,
            joint_states=joint_states,
            joint_dir=joint_dir,
            joint_type="prismatic",
            joint_start=joint_start
        )
        if i % 10 == 0 and logger:
                logger.log(logging.DEBUG, f"[{i}/{num_iterations}] coarse t loss: {loss.item()}")
        
        joint_states_tmp = joint_states.detach().cpu().numpy()
        # 给最开始插入 0
        joint_states_tmp = np.insert(joint_states_tmp, 0, 0)
        joint_start_tmp = joint_start.detach().cpu().numpy()
        joint_dir_tmp = joint_dir.detach().cpu().numpy()
        joint_dir_tmp = joint_dir_tmp / np.linalg.norm(joint_dir_tmp)
        cur_state_dict = {
            "joint_states": joint_states_tmp,
            "joint_start": joint_start_tmp,
            "joint_dir": joint_dir_tmp,
        }
        should_early_stop = scheduler.step(loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        if i % 10 == 0 and logger:
            logger.log(logging.DEBUG, f"\t lr: {cur_lr}")
        
        if should_early_stop:
            if logger:
                logger.log(logging.INFO, "Early stop in coarse_t_estimation")
            break
        
        loss.backward()
        optimizer.step()

    if visualize:
        joint_states = np.copy(scheduler.best_state_dict["joint_states"])
        joint_dir = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_start = np.copy(scheduler.best_state_dict["joint_start"])
        tracks_3d = np.copy(tracks_3d)
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


def coarse_R_from_tracks_3d(tracks_3d, visualize=False, logger=None):
    # 使用一个 frame 的点云估计一整个 normal
    """
    通过所有时间帧的点轨迹估计旋转轴，并通过优化方法求解每帧的旋转角度
    :param tracks_3d: 形状为 (T, M, 3) 的 numpy 数组, T 是时间步数, M 是点的数量
    :return: 旋转轴的单位向量 (3,), 每帧的旋转角度数组 (T,), 以及估计误差 est_loss
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    joint_start, joint_dir = get_init_joint_param(
        tracks_3d=tracks_3d,
        joint_type="revolute"
    )
    joint_states = estimate_joint_state_given_joint_param(
        joint_start=joint_start,
        joint_dir=joint_dir,
        joint_type="revolute",
        tracks_3d=tracks_3d
    )
    
    # 初始化 joint_states 和 joint_start，并设置优化器
    joint_states = torch.from_numpy(joint_states[1:]).float().cuda().requires_grad_() # T-1
    joint_dir = torch.from_numpy(joint_dir).float().cuda().requires_grad_()
    joint_start = torch.from_numpy(joint_start).float().cuda().requires_grad_()

    optimizer = torch.optim.Adam([joint_states, joint_dir, joint_start], lr=1e-2) # 1 cm
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
        loss = loss_function_torch(
            tracks_3d=tracks_3d,
            joint_states=joint_states,
            joint_dir=joint_dir,
            joint_type="revolute",
            joint_start=joint_start
        )
        if i % 10 == 0 and logger:
            logger.log(logging.DEBUG, f"[{i}/{num_iterations}] coarse R loss: {loss.item()}")
        
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
        if i % 10 == 0 and logger:
            logger.log(logging.DEBUG, f"\t lr: {cur_lr}")
        
        if should_early_stop:
            if logger:
                logger.log(logging.INFO, "EARLY STOP")
            break
        
        loss.backward()
        optimizer.step()
        
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


def coarse_R_from_tracks_3d_augmented(tracks_3d, visualize=False, logger=None, num_R_augmented=100):
    # 使用一个 frame 的点云估计一整个 normal
    """
    通过所有时间帧的点轨迹估计旋转轴，并通过优化方法求解每帧的旋转角度
    :param tracks_3d: 形状为 (T, M, 3) 的 numpy 数组, T 是时间步数, M 是点的数量
    :return: 旋转轴的单位向量 (3,), 每帧的旋转角度数组 (T,), 以及估计误差 est_loss
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    # get_init_joint_param 里面对于 joint_dir 进行了修正
    joint_start, joint_dir = get_init_joint_param(
        tracks_3d=tracks_3d,
        joint_type="revolute"
    )
    joint_states = estimate_joint_state_given_joint_param(
        joint_start=joint_start,
        joint_dir=joint_dir,
        joint_type="revolute",
        tracks_3d=tracks_3d
    )
    
    # 在这里对于 joint_dict 进行增广
    # [B, 3] [B, 3]
    batch_joint_start, batch_joint_dir = sample_around_given_joint_param(
        joint_start=joint_start,
        joint_dir=joint_dir,
        num_sample=num_R_augmented
    )
    # 在这里对于 joint_dir 的方向进行修正
    batch_joint_dir = np.array([
        adjust_joint_dir(
            tracks_3d=tracks_3d,
            joint_start=joint_start,
            joint_dir=joint_dir
        ) for (joint_start, joint_dir) in zip(batch_joint_start, batch_joint_dir)
    ])
    # [B, T-1]
    batch_joint_states = np.array([
        estimate_joint_state_given_joint_param(
            joint_start=joint_start,
            joint_dir=joint_dir,
            joint_type="revolute",
            tracks_3d=tracks_3d
        ) for (joint_start, joint_dir) in zip(batch_joint_start, batch_joint_dir)
    ])
    
    # 初始化 joint_states 和 joint_start，并设置优化器
    batch_joint_states = torch.from_numpy(batch_joint_states[:, 1:]).float().cuda().requires_grad_() # B, T-1
    batch_joint_dir = torch.from_numpy(batch_joint_dir).float().cuda().requires_grad_() # B, 3
    batch_joint_start = torch.from_numpy(batch_joint_start).float().cuda().requires_grad_() # B, 3

    optimizer = torch.optim.Adam([batch_joint_states, batch_joint_dir, batch_joint_start], lr=1e-2) # 1 cm
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
        loss, best_loss, best_idx = loss_function_torch_batch(
            tracks_3d=tracks_3d,
            joint_states=batch_joint_states,
            joint_dir=batch_joint_dir,
            joint_type="revolute",
            joint_start=batch_joint_start
        )
        if i % 10 == 0 and logger:
            logger.log(logging.DEBUG, f"[{i}/{num_iterations}] coarse R loss: {loss.item()}")
        
        # NOTE 这里改为存储每次优化所有 batch 最好的那个 joint_dict
        joint_states_tmp = batch_joint_states.detach().cpu().numpy()[best_idx] # B, T-1
        joint_states_tmp = np.insert(joint_states_tmp, 0, 0)
        joint_dir_tmp = batch_joint_dir.detach().cpu().numpy()[best_idx]
        joint_dir_tmp = joint_dir_tmp / np.linalg.norm(joint_dir_tmp)
        joint_start_tmp = batch_joint_start.detach().cpu().numpy()[best_idx]
        cur_state_dict = {
            "joint_states": joint_states_tmp,
            "joint_dir": joint_dir_tmp,
            "joint_start": joint_start_tmp
        }
        should_early_stop = scheduler.step(best_loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        if i % 10 == 0 and logger:
            logger.log(logging.DEBUG, f"\t lr: {cur_lr}")
        
        if should_early_stop:
            if logger:
                logger.log(logging.INFO, "EARLY STOP")
            break
        
        loss.backward()
        optimizer.step()
        
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


def coarse_estimation(tracks_3d, visualize=False, logger=None, num_R_augmented=100):
    """
    根据 tracks3d 估计出初始的 joint 状态, 要求 joint_state 的初始值是0, 且随着轨迹增加
    (如果是旋转的话需要满足右手定则)
    tracks_3d: (T, M, 3)
    """
    t_state_dict, t_est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize, logger)
    # R_state_dict, R_est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize, logger)
    R_state_dict, R_est_loss = coarse_R_from_tracks_3d_augmented(tracks_3d, visualize, logger, num_R_augmented)
    
    if logger:
        logger.log(logging.INFO, f"t_est_loss: {t_est_loss}, R_est_loss: {R_est_loss}")
    # torch.cuda.empty_cache()
    if t_est_loss < R_est_loss:
        if logger:
            logger.log(logging.INFO, "Thus, select as prismatic joint")
        t_state_dict["joint_type"] = "prismatic"
        return t_state_dict
    else:
        if logger:
            logger.log(logging.INFO, "Thus, select as revolute joint")
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
    # test_coarse_R()
    # test_coarse_t_from_tracks_3d()
    # 示例输入 (batch_size=2)
    tracks_3d = np.random.randn(2, 100, 50, 3)  # [B, T, M, 3]
    joint_states = torch.randn(2, 99, requires_grad=True).cuda()
    joint_dir = torch.randn(2, 3, requires_grad=True).cuda()
    joint_start = torch.randn(2, 3, requires_grad=True).cuda()

    # 计算损失
    loss = loss_function_torch_batch(
        tracks_3d, 
        joint_states, 
        joint_dir, 
        "revolute", 
        joint_start
    )
    # loss.backward()