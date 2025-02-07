"""
    利用 depth map 和 sam2 得到的分割结果, 对于 coarse_joint_estimation 得到的结果进行修正
    
    输入:
        depth1, depth2, obj_mask1, obj_mask2, joint_type, joint_axis, joint_state1, joint_state2
    
    输出:
        part_mask1, part_mask2, joint_axis, (joint_state2 - joint_state1)
        
    迭代过程：
        对于 obj_mask1 中的像素进行分类
            假设 static part 和 moving part 对应的变换分别是 T_static 和 T_moving
            对于每个像素，计算其在变换后的三维位置 p_3d 和像素投影 p_2d
            
            正确的变换 => (distance(p_3d, camera_o) >= depth2[p_2d]) and (p_2d in obj_mask2)
            
        对于 obj_mask2 中的像素进行分类
            xxx
            
        找到 moving_mask1 和 moving_mask2 的交集 
            将 moving_mask1 中的像素进行 T_moving 变换, 并得到投影 moving_mask1_project_on2
            求 moving_mask1_project_on2 和 moving_mask2 的交集
            
        用交集的点进行 point-to-plane-ICP 估计出更精确的 joint_axis 和 (joint_state2 - joint_state1)
        
    # TODO: 当有了全局的模型和每一帧的参数后，可以推测出每一帧的 mask, 进而再次给 sam2, 分割出更准的 mask, 进而估计出更准的 joint
"""
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
from embodied_analogy.estimation.icp_custom import point_to_plane_icp
from embodied_analogy.utility.utils import depth_image_to_pointcloud, camera_to_image, reconstruct_mask, visualize_pc
from embodied_analogy.utility.constants import *

def joint_data_to_transform(
    joint_type, # "prismatic" or "revolute"
    joint_axis, # unit vector np.array([3, ])
    joint_state_ref2tgt # joint_state_tgt - joint_state_ref, a constant
):
    # 根据 joint_type 和 joint_axis 和 (joint_state2 - joint_state1) 得到 T_ref2tgt
    T_ref2tgt = np.eye(4)
    if joint_type == "prismatic":
        # coor_tgt = coor_ref + joint_axis * (joint_state_tgt - joint_state_ref)
        T_ref2tgt[:3, 3] = joint_axis * joint_state_ref2tgt
    elif joint_type == "revolute":
        # coor_tgt = coor_ref @ Rref2tgt.T
        Rref2tgt = R.from_rotvec(joint_axis * joint_state_ref2tgt).as_matrix()
        T_ref2tgt[:3, :3] = Rref2tgt
    else:
        assert False, "joint_type must be either prismatic or revolute"
    return T_ref2tgt

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
        Image.fromarray((moving_mask_ref_filtered.astype(np.int32) * 255).astype(np.uint8)).save("moving_mask_ref_filtered.png")
        Image.fromarray((moving_mask_tgt_filtered.astype(np.int32) * 255).astype(np.uint8)).save("moving_mask_tgt_filtered.png")
        
    return pc_ref_filtered, pc_tgt_filtered
    

def fine_joint_estimation(
    K,
    depth_ref, depth_tgt,
    dynamic_mask_ref, dynamic_mask_tgt,
    joint_type, joint_axis, 
    joint_state_ref2tgt,
    visualize=False
):
    """
    对关节参数进行精细估计，基于参考帧和目标帧的深度图及物体掩码，通过点云配准优化关节轴和关节状态。

    参数:
        K (np.ndarray): 相机内参矩阵，形状为 (3, 3)。
        depth_ref (np.ndarray): 参考帧的深度图，形状为 (H, W)。
        depth_tgt (np.ndarray): 目标帧的深度图，形状为 (H, W)。
        dynamic_mask_ref (np.ndarray): 参考帧的动力学掩码，形状为 (H, W), 由0, 1组成。
        dynamic_mask_tgt (np.ndarray): 目标帧的动力学掩码，形状为 (H, W)。
        joint_type (str): 关节类型，支持 "prismatic"（平移关节）或 "revolute"（旋转关节）。
        joint_axis (np.ndarray): 初始估计的关节轴，形状为 (3,)。
        joint_state_ref2tgt (float): 初始估计的参考帧到目标帧的关节状态（平移距离或旋转角度）。
        visualize (bool, 可选): 是否可视化中间步骤，默认值为 False。

    返回:
        fine_joint_axis (np.ndarray): 优化后的关节轴，形状为 (3,)。
        fine_joint_state_ref2tgt (float): 优化后的关节状态，平移距离或旋转角度。

    功能描述:
        基础版本是根据已有的 kinematic_mask 做一个 ICP 估计
        进阶版本是先修正 kinematic_mask, 再做 ICP 估计
    """
    T_ref_to_tgt = joint_data_to_transform(
        joint_type,
        joint_axis,
        joint_state_ref2tgt,
    )
    # TODO: 把这个函数重新写一下，输入的 mask 要求包含初始的 moving 和 static 的估计, 不光输出更准的 joint, 还输出更准的 dynamic_mask
    if False:
        seg_mask_ref = segment_ref_obj_mask(K, depth_ref, depth_tgt, obj_mask_ref, obj_mask_tgt, T_ref_to_tgt)
        seg_mask_tgt = segment_ref_obj_mask(K, depth_tgt, depth_ref, obj_mask_tgt, obj_mask_ref, np.linalg.inv(T_ref_to_tgt))
        
        moving_mask_ref = reconstruct_mask(obj_mask_ref, dynamic_mask_ref == MOVING_LABEL) # H, W
        moving_mask_tgt = reconstruct_mask(obj_mask_tgt, dynamic_mask_tgt == MOVING_LABEL)
        
    moving_mask_ref = (dynamic_mask_ref == MOVING_LABEL)
    moving_mask_tgt = (dynamic_mask_tgt == MOVING_LABEL)
    pc_ref, pc_tgt = find_moving_part_intersection(K, depth_ref, depth_tgt, moving_mask_ref, moving_mask_tgt, T_ref_to_tgt, visualize)
    
    # 如果用于 ICP 的点云数目过少，则直接返回
    if min(len(pc_ref), len(pc_tgt)) < 500:
        return joint_axis, joint_state_ref2tgt
    
    # 然后利用 pc_ref 和 pc_tgt 执行 point-to-plane ICP
    estimated_transform = point_to_plane_icp(pc_ref, pc_tgt, init_transform=T_ref_to_tgt, mode=joint_type, max_iterations=20, tolerance=1e-6)
    
    # 提取数据并返回
    if joint_type == "prismatic":
        fine_joint_state_ref2tgt = np.linalg.norm(estimated_transform[:3, 3])
        fine_joint_axis = estimated_transform[:3, 3] / fine_joint_state_ref2tgt
    elif joint_type == "revolute":
        rotation_matrix = estimated_transform[:3, :3]
        rotation_axis_angle = R.from_matrix(rotation_matrix).as_rotvec()
        fine_joint_state_ref2tgt = np.linalg.norm(rotation_axis_angle)
        fine_joint_axis = rotation_axis_angle / fine_joint_state_ref2tgt
    return fine_joint_axis, fine_joint_state_ref2tgt

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