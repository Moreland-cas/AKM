"""
输入当前的一个 rgb_frame + depth_frame, 输出 joint-state (相对于第一帧参考帧)
还有就是给定 K frame 的 depth + dynamic_mask + joint states + joint_axis_unit + joint type

那输出方式有两种
    一种是基于 优化
    一种是基于 sampling
"""

from embodied_analogy.utility import *
def relocalization(
    K, 
    rgb_frame, depth_frame, 
    ref_dynamic_seq,
    joint_type, 
    joint_axis_unit, 
    ref_joint_states, 
    text_prompt="object",
    positive_points=None,
    negative_points=None,
    use_optimize=True,
    visualize=False
):
    """
    K, 
    rgb_frame: H, W, C
    depth_frame: H, W
    text_prompt: "object"
    positive_points: np.array([N, 2, ])
    negative_points: np.array([N, 2, ])
    ref_dynamic_seq: np.array([N, H, W]), 值为 moving, static 或者 Unknown 中的一个
    joint_type: str, "prismatic" or "revolute"
    joint_axis_unit: np.array([3, ]) , unit vector
    ref_joint_states: np.array([N, ])
    """
    # 首先获取当前帧物体的 mask, 是不是也可以不需要 mask
    depth_valid_mask = depth_frame > 0
    num_ref = len(ref_joint_states)
    if use_optimize:
        initial_query_state = 0
        Tref2querys = [joint_data_to_transform(
            joint_type=joint_type,
            joint_axis_unit=joint_axis_unit,
            joint_state_ref2tgt=initial_query_state - ref_joint_states[i],
        ) for i in range(num_ref)]
        Tref2querys = np.stack(Tref2querys) # T, 4, 4
        
        # 复用求 fine jont estimation 的函数, 但是注意可优化对象变了, 那可能是需要 query_dynamic mask的
    else:
        query_states = [0, 1, 2]
        # given query_state, 计算相对 transform, 即 Tref2query
        for query_state in query_states:
            Tref2querys = [joint_data_to_transform(
                joint_type=joint_type,
                joint_axis_unit=joint_axis_unit,
                joint_state_ref2tgt=query_state - ref_joint_states[i],
            ) for i in range(num_ref)]
            Tref2querys = np.stack(Tref2querys) # T, 4, 4
            
        # 计算所有 ref_frame 的 moving_pc transform 到 query frame 后的 3d 位置和 2d 投影
        
        # 找到 query frame 的 2d 投影位置的观测深度, 看看是不是有 pred_depth >= obs_depth, 计算得分
    pass


if __name__ == "__main__":
    # 测试一下重定位, 可以把 reconstruction 得到的 ref_key_frames 先保留下来
    pass