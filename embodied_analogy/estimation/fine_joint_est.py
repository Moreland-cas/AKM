import os
import torch
import numpy as np
from PIL import Image
from embodied_analogy.estimation.icp_loss import icp_loss_torch
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_image, 
    camera_to_image_torch,
    reconstruct_mask, 
    depth_image_to_pointcloud, 
    joint_data_to_transform,
    napari_time_series_transform,
    compute_normals
)
from embodied_analogy.utility.constants import *

###################### deprecated ######################
def segment_ref_obj_mask(
    K, # 相机内参
    depth_ref, depth_tgt, 
    obj_mask_ref, obj_mask_tgt,
    T_ref2tgt, # (4, 4)
    alpha=0.1,
    visualize=False
):
    # TODO: 摆清你的位置, 你就是个 refine mask 的小函数, 改为 refine mask
    """
    对 obj_mask_ref 中的像素进行分类, 分为 static, moving 和 unknown 三类
    Args:
        obj_mask: 
            要保证这个 mask 里的点的 depth 不为0, 且重投影最好不要落到地面上
        alpha: 
            score_T = alpha * (mask_T_obs - 1) + min(0, depth_T_pred - depth_T_obs)
            score_T 是小于等于 0 的, 且越接近于 0 越好
    """
    # 1) 对 obj_mask_ref 中的像素进行分类
    # 1.1) 首先得到 obj_mask_ref 中为 True 的那些像素点转换到相机坐标系下
    pc_ref = depth_image_to_pointcloud(depth_ref, obj_mask_ref, K) # N, 3
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    
    # 1.2) 分别让这些点按照 T_static (Identity matrix) 或者 T_ref2tgt 运动, 得到变换后的点
    pc_ref_static = pc_ref
    pc_ref_moving = (pc_ref_aug @ T_ref2tgt.T)[:, :3] # N, 3
    
    # 1.3) 将这些点投影，得到新的对应点的像素坐标和深度观测
    uv_static_pred, depth_static_pred = camera_to_image(pc_ref_static, K) # [N, 2], [N, ]
    uv_moving_pred, depth_moving_pred = camera_to_image(pc_ref_moving, K)
    
    # 1.4) 根据像素坐标和深度观测进行打分 
    # TODO：可以把mask的值改为该点离 mask 区域的距离
    # 找到 uv_pred 位置的 mask_obs 和 depth_obs 值, 并且计算得分:
    # score = alpha * (mask_obs - 1) + min(0, depth_pred - depth_obs)
    # 上述得分代表了正确的 Transform 应该满足 depth_pred >= depth_obs 和 mask_obs == True
    uv_static_pred_int = np.floor(uv_static_pred).astype(int)
    mask_static_obs = obj_mask_tgt[uv_static_pred_int[:, 1], uv_static_pred_int[:, 0]] # N
    depth_static_obs = depth_tgt[uv_static_pred_int[:, 1], uv_static_pred_int[:, 0]] # N
    static_score = alpha * (mask_static_obs - 1) + np.minimum(0, depth_static_pred - depth_static_obs) # N
    
    uv_moving_pred_int = np.floor(uv_moving_pred).astype(int)
    mask_moving_obs = obj_mask_tgt[uv_moving_pred_int[:, 1], uv_moving_pred_int[:, 0]]
    depth_moving_obs = depth_tgt[uv_moving_pred_int[:, 1], uv_moving_pred_int[:, 0]]
    moving_score = alpha * (mask_moving_obs - 1) + np.minimum(0, depth_moving_pred - depth_moving_obs) # N
    
    # 1.5）根据 static_score 和 moving_score 将所有点分类为 static, moving 和 unknown 中的一类
    # 得分最大是 0, 如果一方接近 0，另一方很小，则选取接近为 0 的那一类， 否则为 unkonwn
    ref_mask_seg = np.zeros(len(static_score))
    for i in range(len(static_score)):
        if min(abs(static_score[i]), abs(moving_score[i])) > 0.1: # 大于 1dm
            ref_mask_seg[i] = UNKNOWN_LABEL 
        elif max(abs(static_score[i]), abs(moving_score[i])) < 0.001: # 都小于 1mm
            # print(static_score[i], moving_score[i])
            ref_mask_seg[i] = UNKNOWN_LABEL
        elif static_score[i] < moving_score[i]:
            # print(static_score[i], moving_score[i])
            ref_mask_seg[i] = MOVING_LABEL
        else:
            # print(static_score[i], moving_score[i])
            ref_mask_seg[i] = STATIC_LABEL
    if visualize:
        # 使用 napari 进行分类结果的可视化
        pass
    return ref_mask_seg # (N, ), composed of 1, 2， 3

def find_moving_part_intersection(
    K, # 相机内参
    depth_ref, depth_tgt, 
    moving_mask_ref, moving_mask_tgt,
    T_ref2tgt, # (4, 4)
    visualize=False
):
    """
    给定 moving_mask1 和 moving_mask2, 找到投影的交集, 并返回两个 point-cloud, 用于 point-to-plane-ICP
    """
    pc_ref = depth_image_to_pointcloud(depth_ref, moving_mask_ref, K) # N, 3
    # pc_tgt = depth_image_to_pointcloud(depth_tgt, moving_mask_tgt, K) # N, 3
    
    # 首先把 mask_ref 中的点投影到 tgt_frame 中得到一个 projected_mask_ref
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    pc_ref_projected, _ = camera_to_image((pc_ref_aug @ T_ref2tgt.T)[:, :3], K) # N, 2
    
    # 求 pc_ref 变换到 tgt frame 后的投影与 moving_mask_tgt 的交集, 这个交集里的点满足能同时被 ref 和 tgt frame 观测到
    pc_ref_projected_int = np.floor(pc_ref_projected).astype(int) # N, 2
    mask_intersection = moving_mask_tgt[pc_ref_projected_int[:, 1], pc_ref_projected_int[:, 0]] # M
    pc_ref_filtered = pc_ref[mask_intersection] # M, 3
    
    # 将 tgt frame 中 mask_intersection 为 True 的点重投影回 3d 得到 pc_tgt_filtered
    pc_ref_projected_mask = np.zeros_like(moving_mask_tgt, dtype=np.bool_) # H, W
    pc_ref_projected_mask[pc_ref_projected_int[:, 1], pc_ref_projected_int[:, 0]] = True # H, W
    pc_tgt_filtered = depth_image_to_pointcloud(depth_tgt, moving_mask_tgt & pc_ref_projected_mask, K) # M', 3
    
    if visualize:
        # 以一个时序的方式展示 filter 前和 filter 后的点
        moving_mask_ref_filtered = reconstruct_mask(moving_mask_ref, mask_intersection)
        moving_mask_tgt_filtered = pc_ref_projected_mask
        Image.fromarray((moving_mask_ref_filtered.astype(np.int32) * 255).astype(np.uint8)).show()
        Image.fromarray((moving_mask_tgt_filtered.astype(np.int32) * 255).astype(np.uint8)).show()
        
    return pc_ref_filtered, pc_tgt_filtered
###################### deprecated ######################


def classify_unknown():
    # 如果按照深度验证这个必要条件, Tmoving满足但是Tstatic不满足, 那就可以 classify 到 moving, 反之也是
    pass

def filter_dynamic_mask(
    K, # 相机内参
    query_depth, # H, W
    query_dynamic, # H, W
    ref_depths,  # T, H, W
    # ref_dynamics, # T, H, W
    joint_type,
    joint_axis_unit,
    query_state,
    ref_states,
    depth_tolerance=0.01, # 能容忍 1cm 的深度不一致
    visualize=False
):
    """
        根据当前的 joint state
        验证所有的 moving points, 把不确定的 points 标记为 unknown
    """
    Tquery2refs = [
        joint_data_to_transform(
            joint_type,
            joint_axis_unit,
            ref_state - query_state,
    ) for ref_state in ref_states] 
    Tquery2refs = np.array(Tquery2refs) # T, 4, 4
        
    T, H, W = ref_depths.shape
    query_dynamic_updated = query_dynamic.copy()
    
    # 获取当前帧 MOVING_LABEL 的像素坐标
    moving_mask = query_dynamic == MOVING_LABEL
    if not np.any(moving_mask):
        return query_dynamic_updated        

    y, x = np.where(moving_mask) # N
    pc_moving = depth_image_to_pointcloud(query_depth, moving_mask, K)  # (N, 3)
    pc_moving_aug = np.concatenate([pc_moving, np.ones((len(pc_moving), 1))], axis=1)  # (N, 4)
    
    # 批量计算 moving_pc 在其他帧的 3d 坐标
    pc_pred = np.einsum('tij,jk->tik', Tquery2refs, pc_moving_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
    
    # 投影到所有帧
    uv_pred, depth_pred = camera_to_image(pc_pred.reshape(-1, 3), K) # T*N, 2
    uv_pred_int = np.floor(uv_pred.reshape(T, len(pc_moving), 2)).astype(int) # T, N, 2
    depth_pred = depth_pred.reshape(T, len(pc_moving)) # T, N
    
    # 在这里进行一个筛选, 把那些得不到有效 depth_obs 的 moving point 也标记为 Unknown TODO: 这里会不会过于严格
    valid_idx = (uv_pred_int[..., 0] >= 0) & (uv_pred_int[..., 0] < W) & \
                (uv_pred_int[..., 1] >= 0) & (uv_pred_int[..., 1] < H) # T, N
    # 且只有一个时间帧观测不到就认为得不到有效观测
    valid_idx = valid_idx.all(axis=0) # N
    uv_pred_int = uv_pred_int[:, valid_idx] # T, M, 2
    depth_pred = depth_pred[:, valid_idx] # T, M
    
    # TODO: 是否要严格到必须 score_moving > score_static 的点才被保留
    # TODO：获取目标帧的真实深度, 是不是要考虑 depth_ref 等于 0 的情况是否需要拒绝
    
    T_idx = np.arange(T)[:, None]
    depth_obs = ref_depths[T_idx, uv_pred_int[..., 1], uv_pred_int[..., 0]]  # T, M
    
    # 计算误差并更新 dynamic_mask， M, 只要有一帧拒绝，则置为 UNKNOWN
    unknown_mask = (depth_pred + depth_tolerance < depth_obs).any(axis=0)  # M
    query_dynamic_updated[y[valid_idx][unknown_mask], x[valid_idx][unknown_mask]] = UNKNOWN_LABEL
    
    if visualize:
        import napari 
        viewer = napari.view_image((query_dynamic != 0).astype(np.int32), rgb=False)
        viewer.title = "filter current dynamic mask using other frames"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels(query_dynamic.astype(np.int32), name='before filtering')
        viewer.add_labels(query_dynamic_updated.astype(np.int32), name='after filtering')
        
        # 可视化 ref frames
        # viewer.add_labels()
        napari.run()
    
    return query_dynamic_updated  

def filter_dynamic_mask_seq(
    K, # 相机内参
    depth_seq,  # T, H, W
    dynamic_mask_seq, # T, H, W
    joint_type,
    joint_axis_unit,
    joint_states,
    depth_tolerance=0.01, # 能容忍 1cm 的深度不一致
    visualize=False
):
    """
        根据当前的 joint state
        验证所有的 moving points, 把不确定的 points 标记为 unknown
    """
    T, H, W = depth_seq.shape
    dynamic_mask_seq_updated = dynamic_mask_seq.copy()
    
    for i in range(T):
        query_depth = depth_seq[i]
        query_dynamic = dynamic_mask_seq[i]
        query_state = joint_states[i]
        
        other_mask = np.arange(T) != i
        ref_depths = depth_seq[other_mask]
        ref_states = joint_states[other_mask]
        
        query_dynamic_updated = filter_dynamic_mask(
            K, # 相机内参
            query_depth, # H, W
            query_dynamic, # H, W
            ref_depths,  # T, H, W
            # ref_dynamics, # T, H, W
            joint_type,
            joint_axis_unit,
            query_state,
            ref_states,
            depth_tolerance=depth_tolerance, 
            visualize=False
        )
        dynamic_mask_seq_updated[i] = query_dynamic_updated
    
    if visualize:
        import napari 
        viewer = napari.view_image((dynamic_mask_seq != 0).astype(np.int32), rgb=False)
        viewer.title = "filter dynamic mask seq (moving part)"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels(dynamic_mask_seq.astype(np.int32), name='before filtering')
        viewer.add_labels(dynamic_mask_seq_updated.astype(np.int32), name='after filtering')
        napari.run()
    
    return dynamic_mask_seq_updated  # (T, H, W), composed of 1, 2, 3

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


def fine_joint_estimation_seq(
    K,
    depth_seq, 
    dynamic_mask_seq,
    joint_type, 
    joint_axis_unit, # unit vector here
    joint_states,
    max_icp_iters=200, # ICP 最多迭代多少轮
    optimize_joint_axis=True,
    optimize_state_mask=None, # boolean mask to select which states to optimize
    update_dynamic_mask=None,
    lr=3e-4,
    tol=1e-8,
    icp_select_range=0.03,
    visualize=False
):
    """
    主要是计算损失函数的逻辑, 优化没啥难的
    计算损失函数：
        损失函数为所有 (frame_i, frame_j) 的 point-to-plane ICP loss + point-to-point ICP loss
        frame_i 和 frame_j 间的 pc_i 和 pc_j 需要使用验证对齐
        
    如果 optimize_state_mask 或者 update_dynamic_mask 为 None, 则视为全部优化
    """
    # 可视化输入
    if visualize and False:
        import napari 
        viewer = napari.view_image((dynamic_mask_seq != 0).astype(np.int32), rgb=False)
        viewer.title = "visualize input of function fine_joint_estimation_seq"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels(dynamic_mask_seq.astype(np.int32), name='0-query other-ref')
        napari.run()
        
    T = depth_seq.shape[0]
    assert T >= 2
    
    if optimize_state_mask is None:
        optimize_state_mask = np.ones(T, dtype=np.bool_)
    if update_dynamic_mask is None:
        update_dynamic_mask = np.ones(T, dtype=np.bool_)
    
    # 准备 moving mask 数据, 点云数据 和 normal 数据
    pass
    moving_masks = [dynamic_mask_seq[i] == MOVING_LABEL for i in range(T)]
    moving_pcs = [depth_image_to_pointcloud(depth_seq[i], moving_masks[i], K) for i in range(T)] #  [(N, 3), ...], len=T
    normals = [compute_normals(moving_pcs[i]) for i in range(T)]
    
    # 进入 ICP 迭代
    prev_icp_loss = float('inf')
    
    # 设置待优化参数
    axis_params = torch.from_numpy(joint_axis_unit).float().cuda().requires_grad_()
    axis_lr = lr if optimize_joint_axis else 0.0
    
    states_params = [torch.tensor(joint_state).float().cuda().requires_grad_() for joint_state in joint_states]
    state_params_to_optimize = []
    
    for i in range(T):
        param = states_params[i]
        if optimize_state_mask[i]:
            state_lr = lr
        else:
            state_lr = 0.0
        state_params_to_optimize.append({f'params': param, 'lr': state_lr})
            
    # 初始化优化器
    optimizer = torch.optim.Adam([
        {'params': axis_params, 'lr': axis_lr}, 
        *state_params_to_optimize
    ])
    
    # TODO：在这里加一个动态调整学习率的功能
    
    # 生成 (i, j) 对，根据 state_mask 和是否优化 joint_axis 来决定
    ij_pairs = []
    for i in range(T):
        for j in range(T):
            if i >= j:
                continue
            if not optimize_joint_axis and (not optimize_state_mask[i]) and (not optimize_state_mask[j]):
                continue
            ij_pairs.append((i, j))
                    
    for k in range(max_icp_iters):
        # 如果要 update dynamic mask, 就在这里进行
        for l, need_update in enumerate(update_dynamic_mask):
            # 没啥卵用啊我草？？ 如果 initial state 太不准那 filter 出来的结果会很 bug
            if need_update:
                ref_states = np.array([states_param.detach().cpu().item() for states_param in states_params])
                ref_states = ref_states[np.arange(T)!=l]
                dynamic_mask_seq[l] = filter_dynamic_mask(
                    K=K, 
                    query_depth=depth_seq[l], 
                    query_dynamic=dynamic_mask_seq[l], 
                    ref_depths=depth_seq[np.arange(T)!=l],  
                    joint_type=joint_type,
                    joint_axis_unit=axis_params.detach().cpu().numpy(),
                    query_state=states_params[l].detach().cpu().numpy(),
                    ref_states=ref_states,
                    depth_tolerance=0.01, 
                    visualize=visualize
                )
                
                if visualize:
                    import napari 
                    viewer = napari.view_image((dynamic_mask_seq != 0).astype(np.int32), rgb=False)
                    viewer.title = "after first round of dynamic refinement"
                    # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
                    viewer.add_labels(dynamic_mask_seq.astype(np.int32), name='0-query other-ref')
                    napari.run()
                    
                # 还需要更新对应 frame 的 moving_masks 等信息
                moving_masks[l] = dynamic_mask_seq[l] == MOVING_LABEL
                moving_pcs[l] = depth_image_to_pointcloud(depth_seq[l], moving_masks[l], K)
                normals[l] = compute_normals(moving_pcs[l])
        
        # 在这里计算 cur_icp_loss
        cur_icp_loss = 0.0
        
        # 0, 1, 2, ..., T-2, T-1
        for i, j in ij_pairs:
            # 提取需要的数据
            ref_pc = moving_pcs[i]
            tgt_pc = moving_pcs[j]
            target_normals = normals[j]
            
            # 无梯度的计算 Tref2tgt
            Tref2tgt = joint_data_to_transform(
                joint_type, 
                (axis_params / torch.norm(axis_params)).detach().cpu().numpy(), 
                (states_params[j] - states_params[i]).detach().cpu().numpy()
            )
            # 计算 pc_i 和 pc_j 的 intersection_mask
            # pc_ref_mask, pc_tgt_mask = moving_ij_intersection(
            pc_ref_mask, pc_tgt_mask = moving_ij_intersection_torch(
                K,
                ref_pc, tgt_pc,
                moving_masks[i], moving_masks[j],
                Tref2tgt,
                visualize=False,
                i=i,
                j=j
            )
            # 有梯度的计算 ICP loss
            joint_axis_scaled = axis_params / torch.norm(axis_params) * (states_params[j] - states_params[i])
            cur_icp_loss += icp_loss_torch(
                joint_axis_scaled,
                ref_pc[pc_ref_mask],
                tgt_pc[pc_tgt_mask],
                target_normals[pc_tgt_mask],
                loss_type="point_to_plane", # point_to_point
                joint_type=joint_type,
                icp_select_range=icp_select_range
            )
        
        if cur_icp_loss.item() == 0:
            # 如果等于 0, 说明就根本没有有效的数据 pair, 这时候选择更改 icp_range 的参数, 或者直接退出
            print("No valid data pair, exit ICP loop")
            break
            
        # 计算损失变化
        loss_change = abs(cur_icp_loss.item() - prev_icp_loss)
        prev_icp_loss = cur_icp_loss.item()
        print(f"ICP iter_{k}: ", prev_icp_loss)
        # print(f"current estimate state: {states_params[0].detach().cpu().item()}")
        if loss_change < tol:
            break
        
        # otherwise 继续优化
        optimizer.zero_grad()
        cur_icp_loss.backward()
        optimizer.step()

    joint_axis_unit_updated = (axis_params / torch.norm(axis_params)).detach().cpu().numpy()
    joint_states_updated = np.array([states_param.detach().cpu().item() for states_param in states_params])
    
    if visualize:
        # 获取 transform_seq
        transform_seq = [
            joint_data_to_transform(
                joint_type,
                joint_axis_unit,
                cur_state - joint_states[0],
        ) for cur_state in joint_states] # T, 4, 4 (Tfirst2cur)
        transform_seq = np.array(transform_seq)
    
        # 给每个 time_stamp 加上 transformed_first
        pc_ref = moving_pcs[0] # N, 3
        pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1)  # (N, 4)
        
        transform_seq = transform_seq @ np.linalg.inv(transform_seq[0])
        pc_ref_transformed = np.einsum('tij,jk->tik', transform_seq, pc_ref_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
        pc_ref_transformed = napari_time_series_transform(pc_ref_transformed) # T*N, d
        
        transform_seq_updated = np.array([
            joint_data_to_transform(
                joint_type,
                joint_axis_unit_updated,
                cur_state - joint_states_updated[0],
        ) for cur_state in joint_states_updated])
        transform_seq_updated = transform_seq_updated @ np.linalg.inv(transform_seq_updated[0])
        pc_ref_transformed_updated = np.einsum('tij,jk->tik', transform_seq_updated, pc_ref_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
        pc_ref_transformed_updated = napari_time_series_transform(pc_ref_transformed_updated) # T*N, d
        
        moving_pcs = napari_time_series_transform(moving_pcs) # T*N, d
        
        import napari
        viewer = napari.Viewer(ndisplay=3)
        size = 0.002
        viewer.add_points(moving_pcs, size=size, name='moving_pc', opacity=0.8, face_color="blue")
        viewer.add_points(pc_ref_transformed, size=size, name='before icp', opacity=0.8, face_color="red")
        viewer.add_points(pc_ref_transformed_updated, size=size, name='after icp', opacity=0.8, face_color="green")
        viewer.title = "fine joint estimation using ICP"
        napari.run()
        
    return joint_axis_unit_updated, joint_states_updated 


if __name__ == "__main__":
    # 首先读取数据
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
    joint_state_npz = np.load(os.path.join(tmp_folder, "joint_state.npz"))
    joint_states = joint_state_npz["joint_states"]
    
    translation_w = joint_state_npz["translation_w"]
    translation_c = joint_state_npz["translation_c"]
    translation_w_gt = np.array([0, 0, 1.])
    
    Rc2w = np.array([[-4.83808517e-01, -1.60986036e-01,  8.60239983e-01],
       [-8.75173867e-01,  8.89947787e-02, -4.75552976e-01],
       [ 5.21540642e-07, -9.82936144e-01, -1.83947206e-01]])
    translation_c_gt = Rc2w.T @ translation_w_gt
    
    ref_idx, tgt_idx = 0, 23
    
    # 读取 0 和 47 帧的深度图
    depth_ref = np.load(os.path.join(tmp_folder, "depths", f"{ref_idx}.npy")).squeeze() # H, w
    depth_tgt = np.load(os.path.join(tmp_folder, "depths", f"{tgt_idx}.npy")).squeeze()
    
    # 标记深度图观测为 0 的地方
    depth_ref_valid_mask = (depth_ref != 0)
    depth_tgt_valid_mask = (depth_tgt != 0)
    
    # 读取 0 和 47 帧的 obj_mask
    obj_mask_ref = np.array(Image.open(os.path.join(tmp_folder, "sam2_masks", f"{ref_idx}.png"))) # H, W 0, 255
    obj_mask_tgt = np.array(Image.open(os.path.join(tmp_folder, "sam2_masks", f"{tgt_idx}.png")))
    
    obj_mask_ref = (obj_mask_ref == 255)
    obj_mask_tgt = (obj_mask_tgt == 255)
    
    K = np.array([
        [300.,   0., 400.],
        [  0., 300., 300.],
        [  0.,   0.,   1.]]
    )
    
    T_ref_to_tgt = joint_data_to_transform(
        joint_type="prismatic",
        joint_axis=translation_c,
        joint_state_ref=scales[ref_idx],
        joint_state_tgt=scales[tgt_idx]
    )
    # 可视化 tgt frame 的点云, 和 ref frame 的点云经过 transform 后与之的对比，看看估计的怎样
    # from embodied_analogy.utility.utils import visualize_pc
    # points_ref = depth_image_to_pointcloud(depth_ref, obj_mask_ref, K) # N, 3
    # points_ref_transformed = points_ref + translation_c * (joint_state_tgt - joint_state_ref) 
    # colors_ref = np.zeros((len(points_ref), 3))
    # colors_ref[:, 0] = 1
    # visualize_pc(points=points_ref, colors=None)
    
    # points_tgt = depth_image_to_pointcloud(depth_tgt, obj_mask_tgt, K)
    # colors_tgt = np.zeros((len(points_tgt), 3))
    # colors_tgt[:, 1] = 1
    
    # points_concat_for_vis = np.concatenate([points_ref_transformed, points_tgt], axis=0)
    # colors_concat_for_vis = np.concatenate([colors_ref, colors_tgt], axis=0)
    # visualize_pc(points=points_concat_for_vis, colors=colors_concat_for_vis)
    # visualize_pc(points=points_tgt, colors=colors_tgt)
    
    class_mask_ref = segment_ref_obj_mask(K, depth_ref, depth_tgt, obj_mask_ref, obj_mask_tgt, T_ref_to_tgt)
    moving_mask_ref = reconstruct_mask(obj_mask_ref, class_mask_ref == 1) # H, W
    Image.fromarray((moving_mask_ref.astype(np.int32) * 255).astype(np.uint8)).save("mask_ref.png")
    
    class_mask_tgt = segment_ref_obj_mask(K, depth_tgt, depth_ref, obj_mask_tgt, obj_mask_ref, np.linalg.inv(T_ref_to_tgt))
    moving_mask_tgt = reconstruct_mask(obj_mask_tgt, class_mask_tgt == 1)
    Image.fromarray((moving_mask_tgt.astype(np.int32) * 255).astype(np.uint8)).save("mask_tgt.png")
    
    use_intersection = True
    if use_intersection:
        pc_ref, pc_tgt = find_moving_part_intersection(K, depth_ref, depth_tgt, moving_mask_ref, moving_mask_tgt, T_ref_to_tgt, visualize=True)
    else:
        pc_ref = depth_image_to_pointcloud(depth_ref, moving_mask_ref, K)
        pc_tgt = depth_image_to_pointcloud(depth_tgt, moving_mask_tgt, K)
    
    points_concat = np.concatenate([pc_ref, pc_tgt], axis=0)
    colors_ref = np.zeros((len(pc_ref), 3))
    colors_ref[:, 0] = 1 # red
    colors_tgt = np.zeros((len(pc_tgt), 3))
    colors_tgt[:, 1] = 1 # green
    colors_concat = np.concatenate([colors_ref, colors_tgt], axis=0)
    # print(len(pc_ref), len(pc_tgt))
    # visualize_pc(points=points_concat, colors=colors_concat)
    
    # 然后利用 pc_ref 和 pc_tgt 执行 point-to-plane ICP
    from embodied_analogy.estimation.icp_custom import point_to_plane_icp
    import time
    print(np.dot(translation_c_gt, translation_c))
    start = time.time()
    init_transform = np.eye(4)
    estimated_transform = point_to_plane_icp(pc_ref, pc_tgt, init_transform, mode="translation", max_iterations=20, tolerance=1e-6)
    fine_joint_axis = estimated_transform[:3, 3] / np.linalg.norm(estimated_transform[:3, 3])
    end = time.time()
    print(f"without init transform: {end - start}")
    # print(translation_c)
    # print(fine_joint_axis)
    # print(translation_c_gt)
    print(np.dot(translation_c_gt, fine_joint_axis))
    start = time.time()
    init_transform = T_ref_to_tgt
    estimated_transform = point_to_plane_icp(pc_ref, pc_tgt, init_transform, mode="translation", max_iterations=20, tolerance=1e-6)
    fine_joint_axis = estimated_transform[:3, 3] / np.linalg.norm(estimated_transform[:3, 3])
    end = time.time()
    print(f"with init transform: {end - start}")
    print(np.dot(translation_c_gt, fine_joint_axis))