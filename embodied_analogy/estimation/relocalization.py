import torch
import napari 
import numpy as np

from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    joint_data_to_transform_np,
    camera_to_image,
    get_depth_mask
)
from embodied_analogy.estimation.fine_joint_est import fine_estimation
from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.utility.constants import *

def relocalization(
    obj_repr: Obj_repr,
    query_frame: Frame,
    update_query_dynamic=False,
    update_query_contact=False,
    visualize=False
):
    """
    根据 articulated object representation 对 query frame 进行 relocalization
    query_frame:
        需包含 query_depth, query_dynamic
    obj_repr:
        需包含 K, ref_depths, joint_type, joint_dir, ref_joint_states, ref_dynamics
    """
    if query_frame.obj_mask is None:
        print("obj_mask of query_frame is None, run grounded sam first..")
        query_frame.segment_obj(obj_description=obj_repr.obj_description)
    
    if query_frame.dynamic_mask is None:
        print("dynamic_mask of query_frame is None, initializing using obj_mask..")
        query_frame.dynamic_mask = query_frame.obj_mask.astype(np.int32) * MOVING_LABEL
    
    # 从 obj_repr 中读取必要的数据
    K = obj_repr.K
    query_dynamic = query_frame.dynamic_mask
    query_depth = query_frame.depth
    ref_depths = obj_repr.key_frames.get_depth_seq()
    ref_dynamics = obj_repr.key_frames.get_dynamic_seq()
    ref_joint_states = obj_repr.key_frames.get_joint_states()
    assert len(ref_joint_states) == len(obj_repr.key_frames)
        
    # 首先获取当前帧物体的 mask, 是不是也可以不需要 mask
    num_ref = len(obj_repr.key_frames)
    
    # 初始化状态, 通过 rgb 或者 depth 找到最近的图像, 我觉得可以先通过 depth
    best_err = 1e10
    best_matched_idx = -1
    for i in range(num_ref):
        query_depth_mask = get_depth_mask(query_depth, K, obj_repr.Tw2c)
        ref_depth = ref_depths[i]
        ref_depth_mask = get_depth_mask(ref_depth, K, obj_repr.Tw2c)
        
        # 计算 mask 的交集, 对交集中点的深度计算一个观测误差
        inter_mask = query_depth_mask & ref_depth_mask
        delta_depth = (ref_depth - query_depth)**2
        cur_mean_err = delta_depth[inter_mask].mean()
        
        if cur_mean_err < best_err:
            best_err = cur_mean_err
            best_matched_idx = i
    query_state = ref_joint_states[best_matched_idx] 
    
    # 将 query_frame 写进 obj_repr.key_frames, 然后复用 fine_estimation 对初始帧进行优化
    query_frame.joint_state = query_state
    obj_repr.key_frames.frame_list.insert(0, query_frame)
    
    fine_estimation(
        obj_repr=obj_repr,
        opti_joint_dir=False,
        opti_joint_start=False,
        opti_joint_states_mask=np.arange(num_ref+1)==0,
        # 目前不更新 dynamic_mask
        update_dynamic_mask=np.arange(num_ref+1)==-1,
        lr=3e-3,
        visualize=False
    )
    # 然后在这里把 query_frame 从 keyframes 中吐出来
    query_frame = obj_repr.key_frames.frame_list.pop(0)
    
    if update_query_contact:
        # 然后将 initial_frame 中的 contact_3d 迁移到 query_frame 中去, 方便后续生成 query_frame 的抓取 pose
        if obj_repr.initial_frame.joint_state is None:
            print("Initial joint state is None, first reloc init state..")
            reloc_init_frame = relocalization(
                obj_repr,
                query_frame=obj_repr.initial_frame,
                update_query_dynamic=False,
                update_query_contact=False,
                visualize=False
            )
            obj_repr.initial_frame = reloc_init_frame
        
        Tinit2query = joint_data_to_transform_np(
            joint_type=obj_repr.joint_dict["joint_type"],
            joint_dir=obj_repr.joint_dict["joint_dir"],
            joint_start=obj_repr.joint_dict["joint_start"],
            joint_state_ref2tgt=query_frame.joint_state-obj_repr.initial_frame.joint_state
        )
        contact_3d_query = Tinit2query[:3, :3] @ obj_repr.initial_frame.contact3d + Tinit2query[:3, 3] # 3
        query_frame.contact3d = contact_3d_query
        query_frame.contact2d = camera_to_image(contact_3d_query[None], K)[0][0]
    
    # 也就是说把 ref_frame 的 moving part 投影到 query frame 上, 对 query_dynamic 进行一个更新 (其余部分设置为 unknown)
    if update_query_dynamic:
        query_dynamic_zero = np.zeros_like(query_dynamic).astype(np.bool_) # H, W
        for i in range(num_ref):
            ref_moving_pc = depth_image_to_pointcloud(ref_depths[i], ref_dynamics[i]==MOVING_LABEL, K) # N, 3
            Tref2query = joint_data_to_transform_np(
                joint_type=obj_repr.joint_dict["joint_type"],
                joint_dir=obj_repr.joint_dict["joint_dir"],
                joint_start=obj_repr.joint_dict["joint_start"],
                joint_state_ref2tgt=query_frame.joint_state-ref_joint_states[i]
            )
            ref_moving_pc_aug = np.concatenate([ref_moving_pc, np.ones((len(ref_moving_pc), 1))], axis=1) # N, 4
            moving_pc = (ref_moving_pc_aug @ Tref2query.T)[:, :3] # N, 3
            moving_uv, _ = camera_to_image(moving_pc, K) # N, 2
            moving_uv = moving_uv.astype(np.int32)
            tmp_zero = np.zeros_like(query_dynamic)
            tmp_zero[moving_uv[:, 1], moving_uv[:, 0]] = True # H, W
            query_dynamic_zero = query_dynamic_zero | tmp_zero
        
        # 将投影来的点与 obj_mask 进行取交
        obj_mask = query_frame.obj_mask
        # 先把整个物体区域赋值为 UNKNOWN
        query_dynamic_updated = obj_mask * UNKNOWN_LABEL  
        # 再把其中的移动部分赋值为 MOVING_LABEL
        query_dynamic_updated[(obj_mask & query_dynamic_zero).astype(np.bool_)] = MOVING_LABEL
        query_frame.dynamic_mask = query_dynamic_updated
    
    if visualize:
        # 展示 dynamic query 的变化
        viewer = napari.Viewer()
        viewer.title = "relocalization"
        query_frame._visualize(viewer, prefix="query")
        obj_repr._visualize(viewer, prefix="obj_repr")
        print("query frame joint state: ", query_frame.joint_state)
        napari.run()
        
    return query_frame


if __name__ == "__main__":
    # 读取数据
    # obj_index = 44962
    obj_index = 7221
    obj_file_path = f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/reconstruct/recon_data.pkl"
    obj_repr = Obj_repr.load(obj_file_path)
    
    # 开始 relocalization 的部分
    reloc_states = []

    for i in range(len(obj_repr.key_frames)):
        import copy
        obj_repr_tmp = copy.deepcopy(obj_repr)
        query_frame = obj_repr_tmp.key_frames.frame_list.pop(i)
        
        query_frame = relocalization(
            obj_repr_tmp,
            query_frame,
            update_query_dynamic=True,
            update_query_contact=True,
            visualize=False
        )
        reloc_states.append(query_frame.joint_state)
    joint_states = obj_repr.key_frames.get_joint_states()
    print(f"gt states: {joint_states}")
    print("reloc states: ", np.array(reloc_states))
    