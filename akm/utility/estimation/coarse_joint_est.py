import torch
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from akm.utility.utils import (
    napari_time_series_transform,
    fit_plane_normal,
    remove_dir_component,
)
from akm.utility.estimation.scheduler import Scheduler


def random_rotation_matrix(angle_range=60):
    """
    angle_range: The angle of rotation, in degrees
    Generates a random rotation matrix
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
    Perform some sampling near joint_dict and return a series of new joint_dict
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
    Based on tracks_3d , guess the joint_start and joint_dir values.
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
    Because tracks_3d is set to an open joint sequence by default, 
    we need to determine if joint_dir needs to be reversed. 
    This operation only needs to be performed on revolute joints.
    """
    # Make a correction to joint_dir here
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
    The estimation strategy is to abstract each frame into a point and calculate the joint_state based on this point.
    NOTE: This function does not require the revolute object's joint_dir to be in the correct orientation.
    Returns a joint_dict, but with estimated joint_states
    """
    T, M, _ = tracks_3d.shape
    centers = tracks_3d.mean(axis = 1) # T, 3
    
    if joint_type == "prismatic":
        joint_states = np.array([np.dot(centers[i] - centers[0], joint_dir) for i in range(T)])
    else:
        joint_states = np.zeros(T)
        for t in range(1, T):
            # Calculate the point cloud center of the initial frame and the current frame
            center_0 = tracks_3d[0].mean(axis=0)
            center_t = tracks_3d[t].mean(axis=0)
            
            # minus joint_start vector
            diff_0 = center_0 - joint_start
            diff_t = center_t - joint_start
            
            # minus joint_dir vector
            diff_0 = remove_dir_component(diff_0, joint_dir)
            diff_t = remove_dir_component(diff_t, joint_dir)
            
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
        est_loss = torch.mean(torch.norm(reconstructed_tracks - torch.from_numpy(tracks_3d[1:]).cuda(), dim=2))  
        return est_loss
    elif joint_type == "revolute":
        joint_dir = joint_dir / torch.norm(joint_dir)
        skew_v = torch.zeros((3, 3), device="cuda")
        skew_v[0, 1] = -joint_dir[2]
        skew_v[0, 2] = joint_dir[1]
        skew_v[1, 0] = joint_dir[2]
        skew_v[1, 2] = -joint_dir[0]
        skew_v[2, 0] = -joint_dir[1]
        skew_v[2, 1] = joint_dir[0]
        # Calculate the R_reconstructed matrix
        theta = joint_states.unsqueeze(-1).unsqueeze(-1)  # (T-1, 1, 1) 
        # (T-1, 3, 3)
        R_reconstructed = (torch.eye(3, device="cuda") + torch.sin(theta) * skew_v + (1 - torch.cos(theta)) * (skew_v @ skew_v))  
        # translated_initial
        translated_initial = torch.from_numpy(tracks_3d[0]).float().cuda() - joint_start  # N, 3
        reconstructed_track = torch.einsum('t a b, n b -> t n a', R_reconstructed, translated_initial) + joint_start
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
        joint_dir = joint_dir / torch.norm(joint_dir, dim=1, keepdim=True)
        # Get the initial frame [B, 1, M, 3]
        base_frame = tracks_3d_tensor[:, 0:1]
        
        # reconstruct traj: base_frame + joint_states * joint_dir
        # joint_states: [B, T-1] -> [B, T-1, 1, 1]
        # joint_dir: [B, 3] -> [B, 1, 1, 3]
        displacement = joint_states[:, :, None, None] * joint_dir[:, None, None, :] #  [B, T-1, 1, 3]
        reconstructed_tracks = base_frame + displacement #  [B, T-1, M, 3]
        
        # Calculate the L2 error with the subsequent frame [B, T-1, M]
        errors = torch.norm(reconstructed_tracks - tracks_3d_tensor[:, 1:], dim=-1)
        loss = torch.mean(errors)
        # for best selection 
        batchwise_loss = torch.mean(errors, dim=(1, 2)) # B
        best_loss, best_idx = torch.min(batchwise_loss, dim=0)
        return loss, best_loss, best_idx
        
    elif joint_type == "revolute":
        # Normalize joint_dir [B, 3] for each batch
        joint_dir = joint_dir / torch.norm(joint_dir, dim=1, keepdim=True)
        
        # Create a batch skew-symmetric matrix [B, 3, 3]
        skew_v = torch.zeros((B, 3, 3), device=device)
        
        # Filling the skew-symmetric matrix
        skew_v[:, 0, 1] = -joint_dir[:, 2]
        skew_v[:, 0, 2] = joint_dir[:, 1]
        skew_v[:, 1, 0] = joint_dir[:, 2]
        skew_v[:, 1, 2] = -joint_dir[:, 0]
        skew_v[:, 2, 0] = -joint_dir[:, 1]
        skew_v[:, 2, 1] = joint_dir[:, 0]
        
        # Compute the rotation matrix [B, T-1, 3, 3]
        theta = joint_states.unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1]
        eye_batch = torch.eye(3, device=device).repeat(B, 1, 1).unsqueeze(1)  # [B, 1, 3, 3]
        skew_v_batch = skew_v.unsqueeze(1)  # [B, 1, 3, 3]
        
        # Calculate the rotation matrix using the Rodrigues formula
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        R_reconstructed = (
            eye_batch + 
            sin_theta * skew_v_batch + 
            (1 - cos_theta) * torch.matmul(skew_v_batch, skew_v_batch)
        ) # B, T-1, 3, 3
        
        # Calculate the initial position after translation [B, M, 3]
        initial_points = tracks_3d_tensor[:, 0]  # [B, M, 3]
        translated_initial = initial_points - joint_start.unsqueeze(1)  # [B, M, 3]
        
        # Batch rotate point cloud [B, T-1, M, 3] = [B, T-1, 3, 3] @ [B, M, 3]
        # einsum explanation: b: batch, t: time, i,j: matrix dim, m: point index
        reconstructed_track = torch.einsum('b t i j, b m j -> b t m i', 
                                          R_reconstructed, 
                                          translated_initial)
        # Add back the joint starting point [B, T-1, M, 3]
        reconstructed_track += joint_start.unsqueeze(1).unsqueeze(1)
        
        errors = torch.norm(reconstructed_track - tracks_3d_tensor[:, 1:], dim=-1)
        loss = torch.mean(errors)
        # for best selection 
        batchwise_loss = torch.mean(errors, dim=(1, 2)) # B
        best_loss, best_idx = torch.min(batchwise_loss, dim=0)
        return loss, best_loss, best_idx
    
    
def coarse_t_from_tracks_3d(tracks_3d, visualize=False, logger=None):
    """
    Estimate the translation direction by the displacement changes of all time frames and calculate the displacement scalar along this direction for each frame.
        :param tracks_3d: numpy array of shape (T, M, 3), where T is the number of time steps and M is the number of points.
        :return: The average unit vector in the translation direction (3,), the displacement scalar array for each frame (T,)
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

    # Change the initial numpy estimate to a parameter with gradient
    joint_dir = torch.from_numpy(joint_dir).cuda().requires_grad_()
    joint_states = torch.from_numpy(joint_states[1:]).cuda().requires_grad_() # T-1
    joint_start = torch.from_numpy(joint_start).float().cuda().requires_grad_()

    optimizer = torch.optim.Adam([joint_states, joint_dir], lr=1e-2) # 1 cm
    scheduler = Scheduler(
        optimizer=optimizer,
        lr_update_factor=0.5,
        lr_scheduler_patience=3,
        early_stop_patience=10,
    )

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
        # Insert 0 at the beginning
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
        
        joint_start[-1] *= -1
        joint_dir[-1] *= -1
        tracks_3d[..., -1] *= -1
        reconstructed_tracks[..., -1] *= -1
        
        viewer.add_points(napari_time_series_transform(tracks_3d), size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")
        viewer.add_points(napari_time_series_transform(reconstructed_tracks), size=0.01, name='renconstructed tracks 3d', opacity=0.8, face_color="red")

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
    """
    Estimate the rotation axis using the point tracks of all time frames and solve the rotation angle for each frame using an optimization method.
        :param tracks_3d: A numpy array of shape (T, M, 3), where T is the number of time steps and M is the number of points.
        :return: The unit vector of the rotation axis (3,), the array of rotation angles for each frame (T,), and the estimated error est_loss
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
        # NOTE: Since Napari displays a left-handed coordinate system, you need to multiply the z-axis of all 3D data by -1.
        joint_states = np.copy(scheduler.best_state_dict["joint_states"])
        joint_dir = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_start = np.copy(scheduler.best_state_dict["joint_start"])
        
        # Green represents moving part, red represents reconstructed moving part
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
    """
    Estimate the rotation axis using the point tracks of all time frames and solve the rotation angle for each frame using an optimization method.
        :param tracks_3d: A numpy array of shape (T, M, 3), where T is the number of time steps and M is the number of points.
        :return: The unit vector of the rotation axis (3,), the array of rotation angles for each frame (T,), and the estimated error est_loss
    """
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
    
    # get_init_joint_param has corrected the joint_dir
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
    
    # Here we expand the joint_dict
    # [B, 3] [B, 3]
    batch_joint_start, batch_joint_dir = sample_around_given_joint_param(
        joint_start=joint_start,
        joint_dir=joint_dir,
        num_sample=num_R_augmented
    )
    # Correct the direction of joint_dir here
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
    
    # Initialize joint_states and joint_start, and set the optimizer
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
        
        # NOTE Here we store the best joint_dict of all batches optimized each time.
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
        joint_states = np.copy(scheduler.best_state_dict["joint_states"])
        joint_dir = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_start = np.copy(scheduler.best_state_dict["joint_start"])
        
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
    Estimate the initial joint state based on tracks3d. The initial value of joint_state must be 0 and increase with the trajectory.
    (If rotating, the right-hand rule must be followed.)
        tracks_3d: (T, M, 3)
    """
    t_state_dict, t_est_loss = coarse_t_from_tracks_3d(tracks_3d, visualize, logger)
    # R_state_dict, R_est_loss = coarse_R_from_tracks_3d(tracks_3d, visualize, logger)
    R_state_dict, R_est_loss = coarse_R_from_tracks_3d_augmented(tracks_3d, visualize, logger, num_R_augmented)
    
    if logger:
        logger.log(logging.INFO, f"t_est_loss: {t_est_loss}, R_est_loss: {R_est_loss}")
    torch.cuda.empty_cache()
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