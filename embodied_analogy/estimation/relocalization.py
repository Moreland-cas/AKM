"""
输入当前的一个 rgb_frame + depth_frame, 输出 joint-state (相对于第一帧参考帧)
还有就是给定 K frame 的 depth + dynamic_mask + joint states + joint_axis_unit + joint type

那输出方式有两种
    一种是基于 优化
    一种是基于 sampling
"""
import random
from embodied_analogy.estimation.fine_joint_est import fine_joint_estimation_seq
from embodied_analogy.utility import *
def relocalization(
    K, 
    # query_rgb, 
    query_dynamic,
    query_depth, 
    ref_depths, # T, H, W
    joint_type, 
    joint_axis_unit, 
    ref_joint_states, 
    ref_dynamics,
    lr=5e-3,
    tol=1e-7,
    icp_select_range=0.1,
    # text_prompt="object",
    # positive_points=None,
    # negative_points=None,
    visualize=False
):
    # 首先获取当前帧物体的 mask, 是不是也可以不需要 mask
    num_ref = len(ref_joint_states)
    query_state = ref_joint_states[-1] 
    
    # TODO: 在这里补充一个基于 sampling 的初始化方法, 也就是先选取一个比较好的优化初值
    if query_dynamic is None:
        query_dynamic = (query_depth > 0).astype(np.int32) * MOVING_LABEL
    
    # 然后使用其他帧过滤下 query_dynamic
            
    # 复用求 fine jont estimation 的函数, 把需要优化的帧放第一个
    tmp_depths = np.concatenate([query_depth[None], ref_depths], axis=0)
    tmp_joint_states = np.insert(ref_joint_states, 0, query_state)
    tmp_dynamics = np.concatenate([query_dynamic[None], ref_dynamics], axis=0)
    
    _, updated_states = fine_joint_estimation_seq(
        K=K,
        depth_seq=tmp_depths, 
        dynamic_mask_seq=tmp_dynamics,
        joint_type=joint_type, 
        joint_axis_unit=joint_axis_unit, 
        joint_states=tmp_joint_states,
        max_icp_iters=200, 
        optimize_joint_axis=False,
        optimize_state_mask=np.arange(num_ref+1)==0,
        # 目前不更新 dynamic_mask
        update_dynamic_mask=np.arange(num_ref+1)==-1,
        lr=lr,
        tol=tol,
        icp_select_range=icp_select_range,
        visualize=visualize
    )
    query_state_updated = updated_states[0]
    
    if visualize:
        pass
    return query_state_updated


if __name__ == "__main__":
    # 测试一下重定位, 可以把 reconstruction 得到的 ref_key_frames 先保留下来
    pass