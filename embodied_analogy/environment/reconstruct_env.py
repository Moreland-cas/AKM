import numpy as np
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.environment.explore_env import ExploreEnv


class ReconEnv(ExploreEnv):
    def __init__(
            self,
            cfg
        ):        
        super().__init__(cfg=cfg)
        # self.num_initial_pts = recon_cfg["num_initial_pts"]
        self.num_kframes = cfg["num_kframes"]
        self.fine_lr = cfg["fine_lr"]
    
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