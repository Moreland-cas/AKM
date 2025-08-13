import logging
import torch
import numpy as np

from akm.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_image, 
    camera_to_image_torch,
    reconstruct_mask, 
    depth_image_to_pointcloud, 
    joint_data_to_transform_np,
    joint_data_to_transform_torch,
    napari_time_series_transform,
    compute_normals,
)
from akm.utility.constants import *
from akm.utility.estimation.scheduler import Scheduler
from akm.utility.estimation.icp_loss import icp_loss_torch


def moving_ij_intersection(
    K,
    pc_ref,  # N
    pc_tgt,  # M
    moving_mask_ref, moving_mask_tgt,
    Tref2tgt, # (4, 4)
    visualize=False,
    i=-1,
    j=-1
):
    """
    Given moving_mask1 and moving_mask2, find the intersection of the projections and return two point-clouds for point-to-plane-ICP
    """    
    H, W = moving_mask_ref.shape
    Ttgt2ref = np.linalg.inv(Tref2tgt)
    
    # First project the points in mask_ref into tgt_frame to get a projected_mask_ref
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    pc_ref_projected, _ = camera_to_image((pc_ref_aug @ Tref2tgt.T)[:, :3], K) # N, 2
    
    # Find the intersection of the projection of pc_ref transformed to the tgt frame and moving_mask_tgt. 
    # The points in this intersection can be observed by both ref and tgt frames.
    pc_ref_projected_int = np.floor(pc_ref_projected).astype(int) # N, 2
    ref_valid_idx = (pc_ref_projected_int[..., 0] >= 0) & (pc_ref_projected_int[..., 0] < W) & \
                    (pc_ref_projected_int[..., 1] >= 0) & (pc_ref_projected_int[..., 1] < H) # H, W
    pc_ref_projected_int[~ref_valid_idx] = 0
    pc_ref_mask = moving_mask_tgt[pc_ref_projected_int[:, 1], pc_ref_projected_int[:, 0]] # N
    pc_ref_mask[~ref_valid_idx] = False
    
    pc_tgt_aug = np.concatenate([pc_tgt, np.ones((len(pc_tgt), 1))], axis=1) # N, 4
    pc_tgt_projected, _ = camera_to_image((pc_tgt_aug @ Ttgt2ref.T)[:, :3], K) # N, 2
    pc_tgt_projected_int = np.floor(pc_tgt_projected).astype(int) # N, 2
    tgt_valid_idx = (pc_tgt_projected_int[..., 0] >= 0) & (pc_tgt_projected_int[..., 0] < W) & \
                    (pc_tgt_projected_int[..., 1] >= 0) & (pc_tgt_projected_int[..., 1] < H) # H, W
    pc_tgt_projected_int[~tgt_valid_idx] = 0
    pc_tgt_mask = moving_mask_ref[pc_tgt_projected_int[:, 1], pc_tgt_projected_int[:, 0]] # N
    pc_tgt_mask[~tgt_valid_idx] = False

    if visualize:
        import napari
        viewer = napari.view_labels(moving_mask_ref, name=f"ref{i} before filter")
        viewer.add_labels(moving_mask_tgt, name=f'tgt{j} before filter')
        
        moving_mask_ref_filtered = reconstruct_mask(moving_mask_ref, pc_ref_mask)
        viewer.add_labels(moving_mask_ref_filtered, name=f'ref{i} after filter')
        
        moving_mask_tgt_filtered = reconstruct_mask(moving_mask_tgt, pc_tgt_mask)
        viewer.add_labels(moving_mask_tgt_filtered, name=f'tgt{j} after filter')
        
        viewer.title = "find moving mask ij intersection in 2d projection"
        napari.run()
        
    return pc_ref_mask, pc_tgt_mask

@torch.no_grad()
def moving_ij_intersection_torch(
    K,  
    pc_ref,  # N
    pc_tgt,  # M
    moving_mask_ref, moving_mask_tgt,
    Tref2tgt,  # (4, 4)
    visualize=False,
    i=-1,
    j=-1
):
    """
    Given moving_mask1 and moving_mask2, find the intersection of the projections and return two point-clouds for point-to-plane-ICP
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = torch.tensor(K, dtype=torch.float32, device=device)
    pc_ref = torch.tensor(pc_ref, dtype=torch.float32, device=device)
    pc_tgt = torch.tensor(pc_tgt, dtype=torch.float32, device=device)
    moving_mask_ref = torch.tensor(moving_mask_ref, dtype=torch.bool, device=device)
    moving_mask_tgt = torch.tensor(moving_mask_tgt, dtype=torch.bool, device=device)
    Tref2tgt = torch.tensor(Tref2tgt, dtype=torch.float32, device=device)

    H, W = moving_mask_ref.shape
    Ttgt2ref = torch.linalg.inv(Tref2tgt)
    pc_ref_aug = torch.cat([pc_ref, torch.ones((pc_ref.shape[0], 1), device=device)], dim=1)  # N, 4
    pc_ref_projected = (pc_ref_aug @ Tref2tgt.T)[:, :3]  # N, 3

    # Projection to the image plane
    pc_ref_projected, _ = camera_to_image_torch(pc_ref_projected, K)  # N, 2
    pc_ref_projected_int = torch.floor(pc_ref_projected).to(torch.int64)  # N, 2

    # Determine whether the projection is within the image range
    ref_valid_idx = (pc_ref_projected_int[:, 0] >= 0) & (pc_ref_projected_int[:, 0] < W) & \
                    (pc_ref_projected_int[:, 1] >= 0) & (pc_ref_projected_int[:, 1] < H)
    pc_ref_projected_int[~ref_valid_idx] = 0
    pc_ref_mask = moving_mask_tgt[pc_ref_projected_int[:, 1], pc_ref_projected_int[:, 0]]  # N
    pc_ref_mask[~ref_valid_idx] = False

    # Convert pc_tgt to 4xM homogeneous coordinates
    pc_tgt_aug = torch.cat([pc_tgt, torch.ones((pc_tgt.shape[0], 1), device=device)], dim=1)  # M, 4
    pc_tgt_projected = (pc_tgt_aug @ Ttgt2ref.T)[:, :3]  # M, 3

    pc_tgt_projected, _ = camera_to_image_torch(pc_tgt_projected, K)  # M, 2
    pc_tgt_projected_int = torch.floor(pc_tgt_projected).to(torch.int64)  # M, 2

    tgt_valid_idx = (pc_tgt_projected_int[:, 0] >= 0) & (pc_tgt_projected_int[:, 0] < W) & \
                    (pc_tgt_projected_int[:, 1] >= 0) & (pc_tgt_projected_int[:, 1] < H)
    pc_tgt_projected_int[~tgt_valid_idx] = 0
    pc_tgt_mask = moving_mask_ref[pc_tgt_projected_int[:, 1], pc_tgt_projected_int[:, 0]]  # M
    pc_tgt_mask[~tgt_valid_idx] = False

    if visualize:
        import napari
        viewer = napari.view_labels(moving_mask_ref.cpu().numpy(), name=f"ref{i} before filter")
        viewer.add_labels(moving_mask_tgt.cpu().numpy(), name=f'tgt{j} before filter')

        moving_mask_ref_filtered = reconstruct_mask(moving_mask_ref.cpu().numpy(), pc_ref_mask.cpu().numpy())
        viewer.add_labels(moving_mask_ref_filtered, name=f'ref{i} after filter')

        moving_mask_tgt_filtered = reconstruct_mask(moving_mask_tgt.cpu().numpy(), pc_tgt_mask.cpu().numpy())
        viewer.add_labels(moving_mask_tgt_filtered, name=f'tgt{j} after filter')

        viewer.title = "find moving mask ij intersection in 2d projection"
        napari.run()

    return pc_ref_mask.cpu().numpy(), pc_tgt_mask.cpu().numpy()

def fine_estimation(
    K,
    joint_type,
    joint_dir,
    joint_start,
    joint_states,
    depth_seq,
    dynamic_seq,
    opti_joint_dir=True,
    opti_joint_start=True,
    opti_joint_states_mask=None, # boolean mask to select which states to optimize
    # update_dynamic_mask=None,
    lr = 1e-3, # 1mm
    gt_joint_dict=None, # should be in camera frame
    visualize=False,
    logger=None
):
    """
    NOTE: Optimization is performed in the camera coordinate system.
    For the joint_dict of obj_repr and the joint states in the keyframe, select values for optimization based on opti_xxx.
    The loss function is the point-to-point/plane ICP loss for all (frame_i, frame_j).
    """
    T = len(depth_seq)
    assert T >= 2
    
    if opti_joint_states_mask is None:
        opti_joint_states_mask = np.ones(T, dtype=np.bool_)
    
    # Prepare moving mask data, point cloud data and normal data
    moving_masks = [dynamic_seq[i] == MOVING_LABEL for i in range(T)]
    moving_pcs = [depth_image_to_pointcloud(depth_seq[i], moving_masks[i], K) for i in range(T)] #  [(N, 3), ...], len=T
    normals = [compute_normals(moving_pcs[i]) for i in range(T)]
    
    # into ICP iteration
    dir_params = torch.from_numpy(joint_dir).float().cuda().requires_grad_()
    start_params = torch.from_numpy(joint_start).float().cuda().requires_grad_()
    
    dir_lr = lr if opti_joint_dir else 0.0
    start_lr = lr if opti_joint_start else 0.0
    
    states_params = [torch.tensor(joint_state).float().cuda().requires_grad_() for joint_state in joint_states]
    state_params_to_optimize = []
    
    for i in range(T):
        param = states_params[i]
        if opti_joint_states_mask[i]:
            state_lr = lr
        else:
            state_lr = 0.0
        state_params_to_optimize.append({f'params': param, 'lr': state_lr})
            
    optimizer = torch.optim.Adam([
        {'params': dir_params, 'lr': dir_lr}, 
        {'params': start_params, 'lr': start_lr},
        *state_params_to_optimize
    ])
    scheduler = Scheduler(
        optimizer, 
        lr_update_factor=0.5, 
        lr_scheduler_patience=5, 
        early_stop_patience=20
    )
    
    # Generate (i, j) pairs, determined by state_mask and whether to optimize joint_dir
    ij_pairs = []
    for i in range(T):
        for j in range(T):
            if i >= j:
                continue
            # If you do not optimize joint_start or joint_dir, and do not optimize the state
            if (not opti_joint_dir) and (not opti_joint_start) and (not opti_joint_states_mask[i]) and (not opti_joint_states_mask[j]):
                continue
            ij_pairs.append((i, j))
    
    max_icp_iters = 300                    
    for k in range(max_icp_iters):
        cur_icp_loss = 0.0
        # 0, 1, 2, ..., T-2, T-1
        for i, j in ij_pairs:
            ref_pc = moving_pcs[i]
            tgt_pc = moving_pcs[j]
            target_normals = normals[j]
            
            Tref2tgt = joint_data_to_transform_torch(
                joint_type=joint_type, 
                joint_dir=dir_params, 
                joint_start=start_params,
                joint_state_ref2tgt=states_params[j] - states_params[i]
            )
            # Calculate the intersection_mask of pc_i and pc_j
            pc_ref_mask, pc_tgt_mask = moving_ij_intersection_torch(
                K=K,
                pc_ref=ref_pc, 
                pc_tgt=tgt_pc,
                moving_mask_ref=moving_masks[i], 
                moving_mask_tgt=moving_masks[j],
                Tref2tgt=Tref2tgt.detach().cpu().numpy(),
                visualize=False,
                i=i,
                j=j
            )
            # Calculate ICP loss with gradient, input a Tref2tgt instead, change joint_state_to_transform to differentiable
            cur_icp_loss += icp_loss_torch(
                joint_type=joint_type,
                Tref2tgt=Tref2tgt,
                ref_pc=ref_pc[pc_ref_mask],
                tgt_pc=tgt_pc[pc_tgt_mask],
                target_normals=target_normals[pc_tgt_mask],
                loss_type="point_to_plane", # point_to_point
                icp_select_range=0.1
            )
        
        if cur_icp_loss.item() == 0:
            # If it is equal to 0, it means there is no valid data pair at all. 
            # In this case, you can change the icp_range parameter or exit directly.
            logger.log(logging.DEBUG, "No valid data pair, exit ICP loop")
            break
        
        cur_state_dict = {
            "joint_type": joint_type,
            "joint_dir": (dir_params / torch.norm(dir_params)).detach().cpu().numpy(),
            "joint_start": start_params.detach().cpu().numpy(),
            "joint_states": np.array([states_param.detach().cpu().item() for states_param in states_params])
        }
        should_early_stop = scheduler.step(cur_icp_loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        
        if k % 10 == 0:
            logger.log(logging.DEBUG, f"[{k}/{max_icp_iters}] ICP loss: {cur_icp_loss.item()}")
            logger.log(logging.DEBUG, f"\t lr: {cur_lr}")
        
        if should_early_stop:
            logger.log(logging.INFO, "Early stop in fine_estimation")
            break
        
        optimizer.zero_grad()
        cur_icp_loss.backward()
        optimizer.step()
        
    if visualize:
        joint_dir = np.copy(joint_dir)
        joint_states = np.copy(joint_states)
        joint_start = np.copy(joint_start)
        
        joint_dir_updated = np.copy(scheduler.best_state_dict["joint_dir"])
        joint_states_updated = np.copy(scheduler.best_state_dict["joint_states"])
        joint_start_updated = np.copy(scheduler.best_state_dict["joint_start"])
    
        transform_seq = [
            joint_data_to_transform_np(
                joint_type=joint_type,
                joint_dir=joint_dir,
                joint_start=joint_start,
                joint_state_ref2tgt=cur_state - joint_states[0],
        ) for cur_state in joint_states] # T, 4, 4 (Tfirst2cur)
        transform_seq = np.array(transform_seq)
    
        # Add transformed_first to each time_stamp
        pc_ref = moving_pcs[0] # N, 3
        pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1)  # (N, 4)
        
        transform_seq = transform_seq @ np.linalg.inv(transform_seq[0])
        pc_ref_transformed = np.einsum('tij,jk->tik', transform_seq, pc_ref_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
        pc_ref_transformed = napari_time_series_transform(pc_ref_transformed) # T*N, d
        
        transform_seq_updated = np.array([
            joint_data_to_transform_np(
                joint_type=joint_type,
                joint_dir=joint_dir_updated,
                joint_start=joint_start_updated,
                joint_state_ref2tgt=cur_state - joint_states_updated[0],
        ) for cur_state in joint_states_updated])
        transform_seq_updated = transform_seq_updated @ np.linalg.inv(transform_seq_updated[0])
        
        pc_ref_transformed_updated = np.einsum('tij,jk->tik', transform_seq_updated, pc_ref_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
        pc_ref_transformed_updated = napari_time_series_transform(pc_ref_transformed_updated) # T*N, d
        
        moving_pcs = napari_time_series_transform(moving_pcs) # T*N, d
        
        viewer = napari.Viewer(ndisplay=3)
        
        moving_pcs[..., -1] *= -1
        pc_ref_transformed[..., -1] *= -1
        pc_ref_transformed_updated[..., -1] *= -1
        joint_start_updated[-1] *= -1
        joint_dir_updated[-1] *= -1
        
        size = 0.01 / 2
        viewer.add_points(moving_pcs, size=size, name='moving_pc', opacity=0.8, face_color="blue")
        viewer.add_points(pc_ref_transformed, size=size, name='before icp', opacity=0.8, face_color="red")
        viewer.add_points(pc_ref_transformed_updated, size=size, name='after icp', opacity=0.8, face_color="green")
        viewer.title = "fine joint estimation using ICP"
        
        viewer.add_shapes(
            data=np.array([joint_start_updated, joint_start_updated + joint_dir_updated * 0.2]),
            name="joint axis",
            shape_type="line",
            edge_width=0.005,
            edge_color="blue",
            face_color="blue",
        )
        viewer.add_points(
            data=joint_start_updated,
            name="joint start",
            size=0.02,
            face_color="blue",
            border_color="red",
        )
        
        if gt_joint_dict is not None:
            viewer.add_shapes(
                data=np.array([gt_joint_dict["joint_start"], gt_joint_dict["joint_start"] + gt_joint_dict["joint_dir"] * 0.2]),
                name="joint axis",
                shape_type="line",
                edge_width=0.005,
                edge_color="green",
                face_color="blue",
            )
            viewer.add_points(
                data=gt_joint_dict["joint_start"],
                name="joint start",
                size=0.02,
                face_color="green",
                border_color="red",
            )
        napari.run()
    
    torch.cuda.empty_cache()
    return scheduler.best_state_dict