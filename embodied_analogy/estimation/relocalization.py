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
    query_state = ref_joint_states[best_matched_idx] 
    print("best_matched_idx: ", best_matched_idx)
            
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
    visualize = False
    text_prompt = "drawer"

    set_random_seed(79)
    torch.autograd.set_detect_anomaly(True)

    from embodied_analogy.pipeline.process_recorded_data import *
    from embodied_analogy.perception import *
    # 读取数据
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
    recon_data = np.load(tmp_folder + "reconstructed_data.npz")
    
    # 开始 relocalization 的部分
    reloc_states = []

    num_informative_frame_idx = len(recon_data["jonit_states"])
    for i in range(num_informative_frame_idx):
        other_mask = np.arange(num_informative_frame_idx)!=i
        
        # 在这里先生成 query dynamics, 方式是通过 sam 得到物体的 bbox
        initial_bbox, initial_mask = run_grounded_sam(
            rgb_image=recon_data["rgb_seq"][i],
            text_prompt=text_prompt,
            positive_points=None,  # np.array([N, 2])
            negative_points=recon_data["franka_tracks_seq"][i],
            num_iterations=5,
            acceptable_thr=0.9,
            visualize=visualize,
        )
        query_dynamic_initial = initial_mask.astype(np.int32) * MOVING_LABEL
        
        reloc_state = relocalization(
            K=recon_data["K"], 
            query_dynamic=query_dynamic_initial,
            query_depth=recon_data["depth_seq"][i], 
            ref_depths=recon_data["depth_seq"][other_mask], 
            joint_type=recon_data["joint_type"], 
            joint_axis_unit=recon_data["joint_axis"], 
            ref_joint_states=recon_data["jonit_states"][other_mask], 
            ref_dynamics=recon_data["dynamic_mask_seq"][other_mask], 
            lr=5e-3, # 一次估计 0.5 cm?
            tol=1e-7,
            # icp_select_range=0.2,
            icp_select_range=0.2,
            visualize=False
        )
        # print(reloc_state)
        reloc_states.append(reloc_state)
    jonit_states = recon_data["jonit_states"]
    print(f"gt states: {jonit_states}")
    print("reloc states: ", np.array(reloc_states))