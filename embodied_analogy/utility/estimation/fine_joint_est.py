import os
import torch
import numpy as np
from PIL import Image
from embodied_analogy.utility.estimation.icp_loss import icp_loss_torch
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_image, 
    camera_to_image_torch,
    reconstruct_mask, 
    depth_image_to_pointcloud, 
    joint_data_to_transform_np,
    joint_data_to_transform_torch,
    napari_time_series_transform,
    compute_normals,
    initialize_napari
)
initialize_napari()
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.estimation.scheduler import Scheduler

def classify_unknown():
    # 如果按照深度验证这个必要条件, Tmoving满足但是Tstatic不满足, 那就可以 classify 到 moving, 反之也是
    pass


def moving_ij_intersection(
    K, # 相机内参
    pc_ref,  # N
    pc_tgt,  # M
    moving_mask_ref, moving_mask_tgt,
    Tref2tgt, # (4, 4)
    visualize=False,
    i=-1,
    j=-1
):
    """
    给定 moving_mask1 和 moving_mask2, 找到投影的交集, 并返回两个 point-cloud, 用于 point-to-plane-ICP
    """    
    H, W = moving_mask_ref.shape
    Ttgt2ref = np.linalg.inv(Tref2tgt)
    
    # 首先把 mask_ref 中的点投影到 tgt_frame 中得到一个 projected_mask_ref
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    pc_ref_projected, _ = camera_to_image((pc_ref_aug @ Tref2tgt.T)[:, :3], K) # N, 2
    # 求 pc_ref 变换到 tgt frame 后的投影与 moving_mask_tgt 的交集, 这个交集里的点满足能同时被 ref 和 tgt frame 观测到
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
    K,  # 相机内参
    pc_ref,  # N
    pc_tgt,  # M
    moving_mask_ref, moving_mask_tgt,
    Tref2tgt,  # (4, 4)
    visualize=False,
    i=-1,
    j=-1
):
    """
    给定 moving_mask1 和 moving_mask2, 找到投影的交集, 并返回两个 point-cloud, 用于 point-to-plane-ICP
    """
    # 将数据转换为 torch 张量并移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = torch.tensor(K, dtype=torch.float32, device=device)
    pc_ref = torch.tensor(pc_ref, dtype=torch.float32, device=device)
    pc_tgt = torch.tensor(pc_tgt, dtype=torch.float32, device=device)
    moving_mask_ref = torch.tensor(moving_mask_ref, dtype=torch.bool, device=device)
    moving_mask_tgt = torch.tensor(moving_mask_tgt, dtype=torch.bool, device=device)
    Tref2tgt = torch.tensor(Tref2tgt, dtype=torch.float32, device=device)

    # 获取图像的高度和宽度
    H, W = moving_mask_ref.shape

    # 计算反变换 Ttgt2ref
    Ttgt2ref = torch.linalg.inv(Tref2tgt)

    # 将 pc_ref 转换为 4xN 的齐次坐标
    pc_ref_aug = torch.cat([pc_ref, torch.ones((pc_ref.shape[0], 1), device=device)], dim=1)  # N, 4
    pc_ref_projected = (pc_ref_aug @ Tref2tgt.T)[:, :3]  # N, 3

    # 投影到图像平面
    pc_ref_projected, _ = camera_to_image_torch(pc_ref_projected, K)  # N, 2
    pc_ref_projected_int = torch.floor(pc_ref_projected).to(torch.int64)  # N, 2

    # 判断投影是否在图像范围内
    ref_valid_idx = (pc_ref_projected_int[:, 0] >= 0) & (pc_ref_projected_int[:, 0] < W) & \
                    (pc_ref_projected_int[:, 1] >= 0) & (pc_ref_projected_int[:, 1] < H)
    pc_ref_projected_int[~ref_valid_idx] = 0
    pc_ref_mask = moving_mask_tgt[pc_ref_projected_int[:, 1], pc_ref_projected_int[:, 0]]  # N
    pc_ref_mask[~ref_valid_idx] = False

    # 将 pc_tgt 转换为 4xM 的齐次坐标
    pc_tgt_aug = torch.cat([pc_tgt, torch.ones((pc_tgt.shape[0], 1), device=device)], dim=1)  # M, 4
    pc_tgt_projected = (pc_tgt_aug @ Ttgt2ref.T)[:, :3]  # M, 3

    # 投影到图像平面
    pc_tgt_projected, _ = camera_to_image_torch(pc_tgt_projected, K)  # M, 2
    pc_tgt_projected_int = torch.floor(pc_tgt_projected).to(torch.int64)  # M, 2

    # 判断投影是否在图像范围内
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
    visualize=False
):
    """
    NOTE: 在相机坐标系下进行优化的
    对于 obj_repr 的 joint_dict 和 keyframe 中的 joint states 进行优化, 可根据 opti_xxx 有选择的优化部分值
    损失函数为所有 (frame_i, frame_j) 的 point-to-point/plane ICP loss 
    """
    # 读取数据
    T = len(depth_seq)
    assert T >= 2
    
    if opti_joint_states_mask is None:
        opti_joint_states_mask = np.ones(T, dtype=np.bool_)
    # if update_dynamic_mask is None:
    #     update_dynamic_mask = np.ones(T, dtype=np.bool_)
    
    # 准备 moving mask 数据, 点云数据 和 normal 数据
    moving_masks = [dynamic_seq[i] == MOVING_LABEL for i in range(T)]
    moving_pcs = [depth_image_to_pointcloud(depth_seq[i], moving_masks[i], K) for i in range(T)] #  [(N, 3), ...], len=T
    normals = [compute_normals(moving_pcs[i]) for i in range(T)]
    
    # 进入 ICP 迭代
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
            
    # 初始化优化器
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
    
    # 生成 (i, j) 对，根据 state_mask 和是否优化 joint_dir 来决定
    ij_pairs = []
    for i in range(T):
        for j in range(T):
            if i >= j:
                continue
            # 如果不优化 joint_start 或者 joint_dir, 且不优化状态
            if (not opti_joint_dir) and (not opti_joint_start) and (not opti_joint_states_mask[i]) and (not opti_joint_states_mask[j]):
                continue
            ij_pairs.append((i, j))
    
    max_icp_iters = 300                    
    for k in range(max_icp_iters):
        # 如果要 update dynamic mask, 就在这里进行
        # for l, need_update in enumerate(update_dynamic_mask):
            # 没啥卵用啊我草？？ 如果 initial state 太不准那 filter 出来的结果会很 bug
            # if need_update:
            #     ref_states = np.array([states_param.detach().cpu().item() for states_param in states_params])
            #     ref_states = ref_states[np.arange(T)!=l]
            #     dynamic_seq[l] = filter_dynamic_mask(
            #         K=K, 
            #         query_depth=depth_seq[l], 
            #         query_dynamic=dynamic_seq[l], 
            #         ref_depths=depth_seq[np.arange(T)!=l],  
            #         joint_type=joint_type,
            #         joint_dir=dir_params.detach().cpu().numpy(),
            #         joint_start=start_params.detach().cpu().numpy(),
            #         query_state=states_params[l].detach().cpu().numpy(),
            #         ref_states=ref_states,
            #         depth_tolerance=0.01, 
            #         visualize=visualize
            #     )
                
                # if visualize:
                #     import napari 
                #     viewer = napari.view_image((dynamic_seq != 0).astype(np.int32), rgb=False)
                #     viewer.title = "after first round of dynamic refinement"
                #     # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
                #     viewer.add_labels(dynamic_seq.astype(np.int32), name='0-query other-ref')
                #     napari.run()
                    
                # 还需要更新对应 frame 的 moving_masks 等信息
                # moving_masks[l] = dynamic_seq[l] == MOVING_LABEL
                # moving_pcs[l] = depth_image_to_pointcloud(depth_seq[l], moving_masks[l], K)
                # normals[l] = compute_normals(moving_pcs[l])
        
        # 在这里计算 cur_icp_loss
        cur_icp_loss = 0.0
        
        # 0, 1, 2, ..., T-2, T-1
        for i, j in ij_pairs:
            # 提取需要的数据
            ref_pc = moving_pcs[i]
            tgt_pc = moving_pcs[j]
            target_normals = normals[j]
            
            Tref2tgt = joint_data_to_transform_torch(
                joint_type=joint_type, 
                joint_dir=dir_params, 
                joint_start=start_params,
                joint_state_ref2tgt=states_params[j] - states_params[i]
            )
            # 计算 pc_i 和 pc_j 的 intersection_mask
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
            # 有梯度的计算 ICP loss, 改为输入一个 Tref2tgt, 把 joint_state_to_transform 改为可微的
            cur_icp_loss += icp_loss_torch(
                joint_type=joint_type,
                Tref2tgt=Tref2tgt,
                ref_pc=ref_pc[pc_ref_mask],
                tgt_pc=tgt_pc[pc_tgt_mask],
                target_normals=target_normals[pc_tgt_mask],
                loss_type="point_to_plane", # point_to_point
                # loss_type="point_to_point", # point_to_point
                icp_select_range=0.1
            )
        
        if cur_icp_loss.item() == 0:
            # TODO 如果等于 0, 说明就根本没有有效的数据 pair, 这时候选择更改 icp_range 的参数, 或者直接退出
            print("No valid data pair, exit ICP loop")
            break
        
        # 在这里进行 scheduler.step()
        cur_state_dict = {
            "joint_type": joint_type,
            "joint_dir": (dir_params / torch.norm(dir_params)).detach().cpu().numpy(),
            "joint_start": start_params.detach().cpu().numpy(),
            "joint_states": np.array([states_param.detach().cpu().item() for states_param in states_params])
        }
        should_early_stop = scheduler.step(cur_icp_loss.item(), cur_state_dict)
        
        cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        
        if k % 10 == 0:
            print(f"[{k}/{max_icp_iters}] ICP loss:", cur_icp_loss.item())
            print(f"\t lr:", cur_lr)
        
        if should_early_stop:
            print("EARLY STOP")
            break
        
        # otherwise 继续优化
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
    
        # 获取 transform_seq
        transform_seq = [
            joint_data_to_transform_np(
                joint_type=joint_type,
                joint_dir=joint_dir,
                joint_start=joint_start,
                joint_state_ref2tgt=cur_state - joint_states[0],
        ) for cur_state in joint_states] # T, 4, 4 (Tfirst2cur)
        transform_seq = np.array(transform_seq)
    
        # 给每个 time_stamp 加上 transformed_first
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
        
        # 调整坐标系
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


if __name__ == "__main__":
    pass
