"""
    给定两帧铰链物体点云，和初始的 joint estimation, 输出更准的 joint estimation
    
    输入:
        depth1, depth2
        obj_mask1, obj_mask2
        joint_param, delta_joint_state        
    
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
            
        用交集的点进行 ICP 估计出 joint_param 和 delta_joint_state
    
    输出:
        static_mask1, static_mask2
        moving_mask1, moving_mask2
        unknown_mask1, unknown_mask2
        joint_param, delta_joint_state
        
    # TODO: 其实可以加一步, 当我有了全局的模型和每一帧的参数后，可以推测出每一帧的 mask, 进而再次给 sam2, 分割出更准的 mask, 进而估计出更准的 joint
"""
import os
from PIL import Image
import numpy as np
from embodied_analogy.utility.utils import depth_image_to_pointcloud, camera_to_image, reconstruct_mask

def classify_mask(K, depth_ref, depth_tgt, obj_mask_ref, obj_mask_tgt, T_ref2tgt, alpha=1.):
    """
    对 obj_mask_ref 中的像素进行分类, 分为 static, moving 和 unknown 三类
    Args:
        K: 相机内参
        depth: H, W
        obj_mask: H, W (有物体的位置不为 0)
        T_ref2tgt: np.array([4, 4]), 对于 moving part 的运动估计
        alpha: 在计算新的点落入mask和新的点的深度预测大于等于观测值时候的权重
    """
    # 1) 对 obj_mask1 中的像素进行分类
    # 1.1) 首先得到 obj_mask_ref 中为 True 的那些像素点转换到相机坐标系下
    pc_ref = depth_image_to_pointcloud(depth_ref, obj_mask_ref, K) # N, 3
    pc_ref_aug = np.concatenate([pc_ref, np.ones((len(pc_ref), 1))], axis=1) # N, 4
    
    # 1.2) 分别让这些点按照 T_static (Identity matrix) 或者 T_ref2tgt 运动, 得到变换后的点
    pc_tgt_static = pc_ref
    pc_tgt_moving = (pc_ref_aug @ T_ref2tgt.T)[:, :3] # N, 3
    
    # 1.3) 将这些点投影，得到新的对应点的像素坐标和深度观测
    uv_static_pred, depth_static_pred = camera_to_image(pc_tgt_static, K) # [N, 2], [N, ]
    uv_moving_pred, depth_moving_pred = camera_to_image(pc_tgt_moving, K)
    
    # 1.4) 根据像素坐标和深度观测进行打分 TODO：可以把mask的值改为该点离 mask 区域的距离
    # 找到 uv_static_pred 位置的 mask_static_obs 值和 depth_static_obs 值
    # 应该满足 depth_static_pred >= depth_static_obs 和 mask_static_obs == True
    # 计算一个得分，用来反映满足以上条件的程度：alpha * mask_static_obs + min(0, depth_static_pred - depth_static_obs)
    uv_static_pred_int = np.floor(uv_static_pred).astype(int)
    mask_static_obs = obj_mask_tgt[uv_static_pred_int[:, 1], uv_static_pred_int[:, 0]] # N
    depth_static_obs = depth_tgt[uv_static_pred_int[:, 1], uv_static_pred_int[:, 0]] # N
    
    alpha = 0.
    static_score = alpha * (mask_static_obs - 1) + np.minimum(0, depth_static_pred - depth_static_obs) # N
    # static_score = alpha * (mask_static_obs - 1) + (depth_static_pred - depth_static_obs) # N
    
    uv_moving_pred_int = np.floor(uv_moving_pred).astype(int)
    mask_moving_obs = obj_mask_tgt[uv_moving_pred_int[:, 1], uv_moving_pred_int[:, 0]]
    depth_moving_obs = depth_tgt[uv_moving_pred_int[:, 1], uv_moving_pred_int[:, 0]]
    moving_score = alpha * (mask_moving_obs - 1) + np.minimum(0, depth_moving_pred - depth_moving_obs) # N
    # moving_score = alpha * (mask_moving_obs - 1) + (depth_moving_pred - depth_moving_obs) # N
    
    # 1.5）根据 static_score 和 moving_score 将所有点分类为 static, moving 和 unknown 中的一类
    # 得分最大是 0, 如果一方接近 0，另一方很小，则选取接近为 0 的那一类， 否则为 unkonwn
    class_mask = np.zeros(len(static_score))
    for i in range(len(static_score)):
        if depth_static_obs[i] == 1e6 or depth_moving_obs[i] == 1e6:
            class_mask[i] = 0 # 2 for unknown
            continue
        
        if abs(static_score[i]) > 1 and abs(moving_score[i]) > 1:
            class_mask[i] = 0 
            continue
        
        if abs(static_score[i]) < 0.1 and abs(moving_score[i]) < 0.1:
            class_mask[i] = 0 
            continue
        
        if static_score[i] + 0.001 < moving_score[i]:
            class_mask[i] = 1 # 0 for static
        else:
            class_mask[i] = 0 
    return class_mask.astype(np.bool_)
def refine_joint_est(depth1, depth2, obj_mask1, obj_mask2, joint_param, delta_joint_state):
    """
        根据当前的 joint 信息和 obj_mask 信息迭代的进行精度提升
    """

if __name__ == "__main__":
    # 首先读取数据
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
    joint_state_npz = np.load(os.path.join(tmp_folder, "joint_state.npz"))
    scales = joint_state_npz["scales"]
    
    translation_w = joint_state_npz["translation_w"]
    translation_c = joint_state_npz["translation_c"]
    translation_w_gt = np.array([0, 0, 1.])
    
    Rc2w = np.array([[-4.83808517e-01, -1.60986036e-01,  8.60239983e-01],
       [-8.75173867e-01,  8.89947787e-02, -4.75552976e-01],
       [ 5.21540642e-07, -9.82936144e-01, -1.83947206e-01]])
    translation_c_gt = Rc2w.T @ translation_w_gt
    
    ref_idx, tgt_idx = 0, 47
    
    # 读取 0 和 47 帧的深度图
    depth_ref = np.load(os.path.join(tmp_folder, "depths", f"{ref_idx}.npy")).squeeze() # H, w
    depth_tgt = np.load(os.path.join(tmp_folder, "depths", f"{tgt_idx}.npy")).squeeze()
    
    # 标记深度图观测为 0 的地方
    depth_ref_valid_mask = (depth_ref != 0)
    depth_tgt_valid_mask = (depth_tgt != 0)
    
    # 读取 0 和 47 帧的 obj_mask
    obj_mask_ref = np.array(Image.open(os.path.join(tmp_folder, "masks", f"{ref_idx}.png"))) # H, W 0, 255
    obj_mask_tgt = np.array(Image.open(os.path.join(tmp_folder, "masks", f"{tgt_idx}.png")))
    
    obj_mask_ref = (obj_mask_ref == 255)
    obj_mask_tgt = (obj_mask_tgt == 255)
    
    delta_scale = scales[tgt_idx] - scales[ref_idx]
    T_ref_to_tgt = np.eye(4)
    # from 0 to 47, which means coor_47 = coor_0 + translation_c * delta_scale
    T_ref_to_tgt[:3, 3] = translation_c * delta_scale
    
    K = np.array([
        [300.,   0., 400.],
        [  0., 300., 300.],
        [  0.,   0.,   1.]]
    )
    # 可视化 tgt frame 的点云, 和 ref frame 的点云经过 transform 后与之的对比，看看估计的到底有多离谱，你妈的
    from embodied_analogy.utility.utils import visualize_pc
    points_ref = depth_image_to_pointcloud(depth_ref, obj_mask_ref, K) # N, 3
    points_ref_transformed = points_ref + translation_c * delta_scale 
    colors_ref = np.zeros((len(points_ref), 3))
    colors_ref[:, 0] = 1
    # visualize_pc(points=points_ref, colors=None)
    
    points_tgt = depth_image_to_pointcloud(depth_tgt, obj_mask_tgt, K)
    colors_tgt = np.zeros((len(points_tgt), 3))
    colors_tgt[:, 1] = 1
    
    points_concat_for_vis = np.concatenate([points_ref_transformed, points_tgt], axis=0)
    colors_concat_for_vis = np.concatenate([colors_ref, colors_tgt], axis=0)
    if False:
        visualize_pc(points=points_concat_for_vis, colors=colors_concat_for_vis)
    # visualize_pc(points=points_tgt, colors=colors_tgt)
    
    if True:
        class_mask_ref = classify_mask(K, depth_ref, depth_tgt, obj_mask_ref, obj_mask_tgt, T_ref_to_tgt, alpha=1.)
        recon_mask_ref = reconstruct_mask(obj_mask_ref, class_mask_ref)
        Image.fromarray((recon_mask_ref.astype(np.int32) * 255).astype(np.uint8)).save("mask_ref.png")
        
        class_mask_tgt = classify_mask(K, depth_tgt, depth_ref, obj_mask_tgt, obj_mask_ref, np.linalg.inv(T_ref_to_tgt), alpha=1.)
        recon_mask_tgt = reconstruct_mask(obj_mask_tgt, class_mask_tgt)
        Image.fromarray((recon_mask_tgt.astype(np.int32) * 255).astype(np.uint8)).save("mask_tgt.png")