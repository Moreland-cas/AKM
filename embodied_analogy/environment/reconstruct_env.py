import numpy as np
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()
from embodied_analogy.environment.explore_env import ExploreEnv


class ReconEnv(ExploreEnv):
    def __init__(
            self,
            base_cfg={
                "phy_timestep": 1/250.,
                "planner_timestep": None,
                "use_sapien2": True 
            },
            robot_cfg={},
            obj_cfg={
                "index": 44962,
                "scale": 0.8,
                "pose": [1.0, 0., 0.5],
                "active_link": "link_2",
                "active_joint": "joint_2"
            },
            explore_cfg={
                "record_fps": 30,
                "pertubation_distance": 0.1,
                "instruction": "open the drawer",
            },
            recon_cfg={
                "num_initial_pts": 1000,
                "num_kframes": 5,
                "fine_lr": 1e-3,
                "reloc_lr": 3e-3,
            }
        ):        
        super().__init__(
            base_cfg=base_cfg,
            robot_cfg=robot_cfg,
            obj_cfg=obj_cfg,
            explore_cfg=explore_cfg
        )
        self.num_initial_pts = recon_cfg["num_initial_pts"]
        self.num_kframes = recon_cfg["num_kframes"]
        self.fine_lr = recon_cfg["fine_lr"]
        self.reloc_lr = recon_cfg["reloc_lr"]
    
    def reconstruct(self, visualize=False):
        self.obj_repr.reconstruct(
            num_initial_pts=self.num_initial_pts,
            num_kframes=self.num_kframes,
            obj_description=self.obj_description,
            fine_lr=self.fine_lr,
            reloc_lr=self.reloc_lr,
            file_path=None,
            gt_joint_dir_w=None,
            visualize=visualize,
        )
        # TODO: 这里加一个返回是否有 error 的判断
        
    def recon_main(self, visualize=False):
        
        self.explore_main(visualize=visualize)
        
        self.reconstruct(visualize=visualize)
        
    
if __name__ == "__main__":
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        "active_joint": "joint_2"
    }
    obj_index = obj_config["index"]
    
    explore_cfg={
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "instruction": "open the drawer",
        "max_tries": 10
        # instruction="open the microwave",
    }
    recon_cfg={
        "num_initial_pts": 1000,
        "num_kframes": 5,
        "fine_lr": 1e-3
    }
    env = ReconEnv(
        obj_cfg=obj_config,
        explore_cfg=explore_cfg,
        recon_cfg=recon_cfg
    )
    if False:
        from embodied_analogy.representation.obj_repr import Obj_repr
        env.obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
        env.reconstruct(visualize=True)
    else:
        env.recon_main(visualize=False)