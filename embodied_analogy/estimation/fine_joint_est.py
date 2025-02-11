import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from embodied_analogy.estimation.icp_custom import point_to_plane_icp
from embodied_analogy.utility import *


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

def classify_unknown():
    # 如果按照深度验证这个必要条件, Tmoving满足但是Tstatic不满足, 那就可以 classify 到 moving, 反之也是
    pass

def filter_dynamic_mask_seq(
    K, # 相机内参
    depth_seq,  # T, H, W
    dynamic_mask_seq, # T, H, W
    transform_seq, # (T, 4, 4) 把 frame_0 作为 world_frame, Tw2i
    visualize=False
):
    """
        根据当前的 joint state
        验证所有的 moving points, 把不确定的 points 标记为 unknown
    """
    T, H, W = depth_seq.shape
    dynamic_mask_seq_updated = dynamic_mask_seq.copy()
    
    for i in range(T):
        # 获取当前帧 MOVING_LABEL 的像素坐标
        moving_mask = dynamic_mask_seq[i] == MOVING_LABEL
        if not np.any(moving_mask):
            continue
        
        y, x = np.where(moving_mask) # N
        pc_moving = depth_image_to_pointcloud(depth_seq[i], moving_mask, K)  # (N, 3)
        pc_moving_aug = np.concatenate([pc_moving, np.ones((len(pc_moving), 1))], axis=1)  # (N, 4)
        
        # 批量计算所有其他帧的转换
        T_i_to_all = transform_seq @ np.linalg.inv(transform_seq[i])  # (T, 4, 4)
        pc_pred = np.einsum('tij,jk->tik', T_i_to_all, pc_moving_aug.T).transpose(0, 2, 1)[:, :, :3] # T, N, 3
        
        # 投影到所有帧
        uv_pred, depth_pred = camera_to_image(pc_pred.reshape(-1, 3), K)  
        uv_pred = uv_pred.reshape(T, len(pc_moving), 2) # T, N, 2
        depth_pred = depth_pred.reshape(T, len(pc_moving)) # T, N
        
        uv_pred_int = np.floor(uv_pred).astype(int) # T, N, 2
        # TODO:考虑超出图像边界的情况
        # valid_idx = (uv_pred_int[..., 0] >= 0) & (uv_pred_int[..., 0] < W) & \
        #             (uv_pred_int[..., 1] >= 0) & (uv_pred_int[..., 1] < H)
        
        # valid_uv = uv_pred_int[valid_idx] # M, 2
        # depth_pred_valid = depth_pred[valid_idx] # M
        # TODO: 是否要严格到必须 score_moving > score_static 的点才被保留
        
        # 获取目标帧的真实深度
        T_idx = np.arange(T)[:, None]
        depth_obs = depth_seq[T_idx, uv_pred_int[..., 1], uv_pred_int[..., 0]]  # T, M
        
        # 计算误差并更新 dynamic_mask
        depth_tolerance = 0.01
        update_to_unknown = (depth_pred + depth_tolerance < depth_obs).any(axis=0)  # M, 只要有一帧拒绝，则置为 UNKNOWN
        dynamic_mask_seq_updated[i, y[update_to_unknown], x[update_to_unknown]] = UNKNOWN_LABEL
    
    if visualize:
        import napari 
        viewer = napari.view_image((dynamic_mask_seq != 0).astype(np.int32), rgb=False)
        viewer.title = "filter dynamic mask seq (moving part)"
        # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
        viewer.add_labels(dynamic_mask_seq.astype(np.int32), name='before filtering')
        viewer.add_labels(dynamic_mask_seq_updated.astype(np.int32), name='after filtering')
        napari.run()
    
    return dynamic_mask_seq_updated  # (T, H, W), composed of 1, 2, 3


def intersect_moving_part_in_2d(
    K, # 相机内参
    depth_ref, depth_tgt, 
    moving_mask_ref, moving_mask_tgt,
    Tref2tgt, # (4, 4)
    visualize=False
):
    """
    给定 moving_mask1 和 moving_mask2, 找到投影的交集, 并返回两个 point-cloud, 用于 point-to-plane-ICP
    """
    pc_ref = depth_image_to_pointcloud(depth_ref, moving_mask_ref, K) # N, 3
    # pc_tgt = depth_image_to_pointcloud(depth_tgt, moving_mask_tgt, K) # N, 3
    
    # 首先把 mask_ref 中的点投影到 tgt_frame 中得到一个 projected_mask_ref
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    pc_ref_projected, _ = camera_to_image((pc_ref_aug @ Tref2tgt.T)[:, :3], K) # N, 2
    
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
        import napari
        viewer = napari.view_labels(moving_mask_ref, name="ref before filter")
        viewer.title = "find moving mask ij intersection in 2d projection"
        viewer.add_labels(moving_mask_ref_filtered, name='ref after filter')
        viewer.add_labels(moving_mask_tgt, name='tgt before filter')
        viewer.add_labels(moving_mask_tgt_filtered, name='tgt after filter')
        
    return pc_ref_filtered, pc_tgt_filtered
    

def fine_joint_estimation_seq(
    K,
    depth_seq, 
    dynamic_mask_seq,
    joint_type, 
    joint_axis, 
    joint_states,
    visualize=False
):
    """
    主要是计算损失函数的逻辑, 优化没啥难的
    计算损失函数：
        损失函数为所有 (frame_i, frame_j) 的 point-to-plane ICP loss + point-to-point ICP loss
        frame_i 和 frame_j 间的 pc_i 和 pc_j 需要使用验证对齐
        
    """
    # 首先对于 moving_part 进行 K 帧的验证（利用 joint states）, 去除那些可能有问题的 part, 至此 moving mask 不动了
    
    # 更新 joint states, 优化 N 次
    #   sum of all (i, j): 对于 moving_i 和 moving_j, 先求交集, 再计算两种 ICP loss (需要 joint states 离散的计算分配关系)
    #   然后调用 torch_minimize 更新 joint states
        
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