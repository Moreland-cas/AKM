import numpy as np
from embodied_analogy.utility.utils import (
    initialize_napari,
    joint_data_to_transform_np
)
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.environment.explore_env import ExploreEnv
from embodied_analogy.representation.basic_structure import Frame

class ReconEnv(ExploreEnv):
    def __init__(
            self,
            cfg
        ):        
        super().__init__(cfg=cfg)
        self.num_kframes = cfg["num_kframes"]
        self.fine_lr = cfg["fine_lr"]
    
    def update_cur_frame(self, visualize=False):
        self.base_step()
        cur_frame = self.capture_frame()
        cur_frame = self.obj_repr.reloc(
            query_frame=cur_frame,
            reloc_lr=self.reloc_lr,
            visualize=visualize
        )
        self.cur_state = cur_frame.joint_state
        self.cur_frame = cur_frame
        
    def ref_ph_to_tgt(self, ref_frame: Frame, tgt_frame: Frame):
        """
        将 ref_frame 中的 panda_hand grasp_pose 转换到 target_frame 中去
        
        NOTE: 
            由于现在的抓取模块不是很强, 所以需要这个函数, 也就是说我们 transfer 的不是 contact_3d, 
            而是 explore 阶段已经证实比较好的一个 panda_hand grasp pose
        """
        Tph2w_ref = ref_frame.Tph2w
        Tph2c_ref = self.obj_repr.Tw2c @ Tph2w_ref
        
        # Tref2tgt 是 camera 坐标系下的一个变换
        Tref2tgt_c = joint_data_to_transform_np(
            joint_type=self.obj_repr.fine_joint_dict["joint_type"],
            joint_dir=self.obj_repr.fine_joint_dict["joint_dir"],
            joint_start=self.obj_repr.fine_joint_dict["joint_start"],
            joint_state_ref2tgt=tgt_frame.joint_state-ref_frame.joint_state
        )
        
        Tph2c_tgt = Tref2tgt_c @ Tph2c_ref
        Tph2w_tgt = np.linalg.inv(self.obj_repr.Tw2c) @ Tph2c_tgt
        tgt_frame.Tph2w = Tph2w_tgt

    def recon_stage(self, load_path=None, save_path=None, visualize=False):
        if load_path is not None:
            self.obj_repr = Obj_repr.load(load_path)
        
        self.obj_repr.reconstruct(
            # num_initial_pts=self.num_initial_pts,
            num_kframes=self.num_kframes,
            obj_description=self.obj_description,
            fine_lr=self.fine_lr,
            file_path=None,
            gt_joint_dir_w=None,
            visualize=visualize,
        )
        
        if save_path is not None:
            self.obj_repr.save(save_path)
                    
    
if __name__ == "__main__":
    cfg = {
        "num_kframes": 5,
        "fine_lr": 1e-3,
    }
    env = ReconEnv(cfg)
    
    env.recon_stage(
        load_path=f"/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore_4_11/40147_1_prismatic/explore/obj_repr.npy",    
        save_path=None,
        evaluate=True,
        visualize=False
    )