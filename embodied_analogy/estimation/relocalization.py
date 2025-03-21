import torch
import napari 
import numpy as np

from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    joint_data_to_transform_np,
    camera_to_image
)
from embodied_analogy.estimation.fine_joint_est import fine_joint_estimation_seq
from embodied_analogy.perception.grounded_sam import run_grounded_sam
from embodied_analogy.utility.constants import *

def relocalization(
    K, 
    query_rgb, 
    query_depth, 
    ref_depths, # T, H, W
    joint_type, 
    joint_dir, 
    ref_joint_states, 
    ref_dynamics,
    lr=5e-3,
    icp_select_range=0.1,
    obj_description="object",
    negative_points=None, # N, 2
    visualize=False
):
    # 相较与 _relocalization, 加入了 grounded sam 的部分    
    _, obj_mask = run_grounded_sam(
        rgb_image=query_rgb,
        obj_description=obj_description,
        positive_points=None, 
        negative_points=negative_points, # N, 2
        num_iterations=3,
        acceptable_thr=0.9,
        visualize=visualize
    )
    query_dynamic = obj_mask.astype(np.int32) * MOVING_LABEL
        
    query_state_updated, query_dynamic_updated = _relocalization(
        K, 
        query_dynamic,
        query_depth, 
        ref_depths, 
        joint_type, 
        joint_dir, 
        ref_joint_states, 
        ref_dynamics,
        lr=lr,
        icp_select_range=icp_select_range,
        visualize=visualize
    )
    return query_state_updated, obj_mask, query_dynamic_updated

def _relocalization(
    K, 
    query_dynamic,
    query_depth, 
    ref_depths, # T, H, W
    joint_type, 
    joint_dir, 
    ref_joint_states, 
    ref_dynamics,
    lr=5e-3,
    icp_select_range=0.1,
    visualize=False
):
    if query_dynamic is None:
        query_dynamic = (query_depth > 0).astype(np.int32) * MOVING_LABEL
        
    # 首先获取当前帧物体的 mask, 是不是也可以不需要 mask
    num_ref = len(ref_joint_states)
    
    # 初始化状态, 通过 rgb 或者 depth 找到最近的图像, 我觉得可以先通过 depth
    best_err = 1e10
    best_matched_idx = -1
    for i in range(num_ref):
        query_depth_mask = query_dynamic > 0 # 除去背景的部分就是物体的所有点
        ref_depth = ref_depths[i]
        ref_depth_mask = ref_dynamics[i] > 0
        
        # 计算 mask 的交集, 对交集中点的深度计算一个观测误差
        inter_mask = query_depth_mask & ref_depth_mask
        delta_depth = (ref_depth - query_depth)**2
        cur_mean_err = delta_depth[inter_mask].mean()
        
        if cur_mean_err < best_err:
            best_err = cur_mean_err
            best_matched_idx = i
    # best_matched_idx = 1
    query_state = ref_joint_states[best_matched_idx] 
    print("best_matched_idx: ", best_matched_idx)
            
    # 复用求 fine jont estimation 的函数, 把需要优化的帧放第一个
    tmp_depths = np.concatenate([query_depth[None], ref_depths], axis=0)
    tmp_joint_states = np.insert(ref_joint_states, 0, query_state)
    tmp_dynamics = np.concatenate([query_dynamic[None], ref_dynamics], axis=0)
    
    _, updated_states = fine_joint_estimation_seq(
        K=K,
        depth_seq=tmp_depths, 
        dynamic_seq=tmp_dynamics,
        joint_type=joint_type, 
        joint_dir=joint_dir, 
        joint_states=tmp_joint_states,
        max_icp_iters=200, 
        optimize_joint_dir=False,
        optimize_state_mask=np.arange(num_ref+1)==0,
        # 目前不更新 dynamic_mask
        update_dynamic_mask=np.arange(num_ref+1)==-1,
        lr=lr,
        icp_select_range=icp_select_range,
        visualize=visualize
    )
    query_state_updated = updated_states[0]

    # 在这里对于 query_dynamic 进行更新
    # 也就是说把 ref_frame 的 moving part 投影到 query frame 上, 对 query_dynamic 进行一个更新 (其余部分设置为 unknown)
    query_dynamic_zero = np.zeros_like(query_dynamic).astype(np.bool_) # H, W
    for i in range(num_ref):
        ref_moving_pc = depth_image_to_pointcloud(ref_depths[i], ref_dynamics[i]==MOVING_LABEL, K) # N, 3
        Tref2query = joint_data_to_transform_np(
            joint_type=joint_type,
            joint_dir=joint_dir,
            joint_state_ref2tgt=query_state_updated-ref_joint_states[i]
        )
        ref_moving_pc_aug = np.concatenate([ref_moving_pc, np.ones((len(ref_moving_pc), 1))], axis=1) # N, 4
        moving_pc = (ref_moving_pc_aug @ Tref2query.T)[:, :3] # N, 3
        moving_uv, _ = camera_to_image(moving_pc, K) # N, 2
        moving_uv = moving_uv.astype(np.int32)
        tmp_zero = np.zeros_like(query_dynamic)
        tmp_zero[moving_uv[:, 1], moving_uv[:, 0]] = True # H, W
        query_dynamic_zero = query_dynamic_zero | tmp_zero
    
    obj_mask = (query_dynamic > 0)
    query_dynamic_updated = obj_mask * UNKNOWN_LABEL # H, W, 一些为 True 的地方 
    query_dynamic_updated[(obj_mask & query_dynamic_zero).astype(np.bool_)] = MOVING_LABEL
    
    if visualize:
        # 展示 dynamic query 的变化
        viewer = napari.view_image((query_dynamic != 0).astype(np.int32), rgb=False)
        viewer.title = "relocalization"
        viewer.add_labels(query_dynamic.astype(np.int32), name='initial query dynamic')
        viewer.add_labels(query_dynamic_updated.astype(np.int32), name='filtered query dynamic')
        napari.run()
    return query_state_updated, query_dynamic_updated


if __name__ == "__main__":
    visualize = False
    obj_description = "drawer"

    torch.autograd.set_detect_anomaly(True)

    # from embodied_analogy.perception import *
    # 读取数据
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/reconstruct/"
    obj_repr = np.load(tmp_folder + "obj_repr.npz")
    
    # 开始 relocalization 的部分
    reloc_states = []

    num_informative_frame_idx = len(obj_repr["joint_states"])
    for i in range(num_informative_frame_idx):
        other_mask = np.arange(num_informative_frame_idx)!=i
        
        # 在这里先生成 query dynamics, 方式是通过 sam 得到物体的 bbox
        from embodied_analogy.perception.grounded_sam import run_grounded_sam
        initial_bbox, initial_mask = run_grounded_sam(
            rgb_image=obj_repr["rgb_seq"][i],
            obj_description=obj_description,
            positive_points=None,  # np.array([N, 2])
            negative_points=None,
            num_iterations=5,
            acceptable_thr=0.9,
            visualize=visualize,
        )
        # query_dynamic_initial = initial_mask.astype(np.int32) * MOVING_LABEL
        # query_state_updated, obj_mask, query_dynamic_updated
        reloc_state, _, _ = relocalization(
            K=obj_repr["K"], 
            # query_dynamic=query_dynamic_initial,
            query_rgb=obj_repr["rgb_seq"][i],
            query_depth=obj_repr["depth_seq"][i], 
            ref_depths=obj_repr["depth_seq"][other_mask], 
            joint_type=obj_repr["joint_type"], 
            joint_dir=obj_repr["joint_dir_c"], 
            ref_joint_states=obj_repr["joint_states"][other_mask], 
            ref_dynamics=obj_repr["dynamic_seq"][other_mask], 
            lr=1e-3, # 一次估计 1 mm
            icp_select_range=0.2,
            visualize=visualize
        )
        reloc_states.append(reloc_state)
    joint_states = obj_repr["joint_states"]
    print(f"gt states: {joint_states}")
    print("reloc states: ", np.array(reloc_states))