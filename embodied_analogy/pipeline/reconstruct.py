import numpy as np

from embodied_analogy.utility.utils import (
    initialize_napari,   
    set_random_seed,
)

initialize_napari()
from embodied_analogy.utility.constants import *
set_random_seed(SEED)

def reconstruct(
    obj_repr,
    num_initial_pts=1000,
    num_kframes=5,
    obj_description="drawer",
    file_path=None,
    gt_joint_dir=None,
    visualize=True,
):
    """
        读取 exploration 阶段获取的视频数据, 进而对物体结构进行恢复
        返回一个 obj_repr, 包含物体表示和 joint 估计
    """
    # 根据物体初始状态的图像, 得到一些初始跟踪点, initial_uvs
    obj_repr.frames[0].segment_obj(
        obj_description=obj_description,
        post_process_mask=True,
        filter=True,
        visualize=visualize
    )
    # 在 initial_bbox 内均匀采样
    obj_repr.frames[0].sample_points(num_points=num_initial_pts, visualize=visualize)
    """
        对于 initial_uvs 进行追踪得到 tracks2d 
    """
    obj_repr.frames.track_points(visualize=visualize)
    """
        将 tracks2d 升维到 tracks3d, 并在 3d 空间对 tracks 进行聚类
    """
    # 首先将 tracks2d 升维到 tracks3d
    obj_repr.frames.track2d_to_3d(visualize=visualize)
    # 在 3d 空间对 tracks 进行聚类
    obj_repr.frames.cluster_track3d(visualize=visualize)
    """
        coarse joint estimation with tracks3d
    """
    obj_repr.coarse_joint_estimation(visualize=visualize)
    # 初始化 kframes
    obj_repr.initialize_kframes(num_kframes=num_kframes, save_memory=True)
    """
        对 kframes 进行完整的 sam2 
    """
    obj_repr.kframes.segment_obj(obj_description=obj_description, visualize=visualize)
    """
        根据 tracks2d 和 obj_mask_seq 得到 dynamic_seq
    """
    obj_repr.kframes.classify_dynamics(
        filter=True,
        joint_dict=obj_repr.coarse_joint_dict,
        visualize=visualize
    )
    """
        根据 dynamic_seq 中的 moving_part, 利用 ICP 估计出精确的 joint params
    """
    obj_repr.fine_joint_estimation(lr=1e-3, visualize=visualize)
        
    if file_path is not None:
        obj_repr.visualize()
        obj_repr.save(file_path)
    
    if gt_joint_dir is not None:
        print(f"\tgt axis: {gt_joint_dir}")

        dot_before = np.dot(gt_joint_dir, joint_dir_w)
        print(f"\tbefore: {np.degrees(np.arccos(dot_before))}")
        print("\tjoint axis: ", joint_dir_w)
        
        print("\tjoint states: ", joint_states[kf_idx])

        dot_after = np.dot(gt_joint_dir, joint_dir_w_updated)
        print(f"\tafter : {np.degrees(np.arccos(dot_after))}")
        print("\tjoint axis: ", joint_dir_w_updated)
        print("\tjoint states: ", joint_states_updated)
    

if __name__ == "__main__":
    from embodied_analogy.representation.obj_repr import Obj_repr
    # obj_idx = 7221
    obj_idx = 44962
    obj_repr_path = f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/explore/explore_data.pkl"
    obj_repr_data = Obj_repr.load(obj_repr_path)
    # obj_repr_data.frames.frame_list.reverse()
    
    reconstruct(
        obj_repr=obj_repr_data,
        num_initial_pts=1000,
        num_kframes=5,
        visualize=False,
        # gt_joint_dir=np.array([-1, 0, 0]),
        # gt_joint_dir=np.array([0, 0, 1]),
        gt_joint_dir=None,
        file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/reconstruct/recon_data.pkl"
        # file_path = None
    )
