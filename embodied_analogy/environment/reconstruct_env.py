import numpy as np
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr
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
            task_cfg={
                "instruction": "open the drawer",
                "obj_description": "drawer",
                "obj_cfg": {
                    "index": 44962,
                    "scale": 0.8,
                    "pose": [1.0, 0., 0.5],
                    "active_link": "link_2",
                    "active_joint": "joint_2"
                },
            },
            explore_cfg={
                "record_fps": 30,
                "pertubation_distance": 0.1,
                "max_tries": 10,
            },
            recon_cfg={
                "num_initial_pts": 1000,
                "num_kframes": 5,
                "fine_lr": 1e-3,
            }
        ):        
        super().__init__(
            base_cfg=base_cfg,
            robot_cfg=robot_cfg,
            task_cfg=task_cfg,
            explore_cfg=explore_cfg
        )
        self.num_initial_pts = recon_cfg["num_initial_pts"]
        self.num_kframes = recon_cfg["num_kframes"]
        self.fine_lr = recon_cfg["fine_lr"]
    
    def recon_stage(self, load_path=None, save_path=None, visualize=False):
        if load_path is not None:
            self.obj_repr.load(load_path)
        
        self.obj_repr.reconstruct(
            num_initial_pts=self.num_initial_pts,
            num_kframes=self.num_kframes,
            obj_description=self.obj_description,
            fine_lr=self.fine_lr,
            file_path=None,
            gt_joint_dir_w=None,
            visualize=visualize,
        )
        
        if save_path is not None:
            self.obj_repr.save(save_path)
            
        # TODO: 这里加一个返回是否有 error 的判断
        
    
if __name__ == "__main__":
    explore_cfg={
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "max_tries": 10,
        "update_sigma": 0.05
    }
    task_cfg={
        "instruction": "open the drawer",
        "obj_description": "drawer",
        "delta_state": 0.1,
        "obj_cfg": {
            "asset_path": "/home/zby/Programs/VideoTracking-For-AxisEst/downloads/dataset/one_drawer_cabinet/48878_link_0", 
            "scale": 0.8,
            "active_link_name": "link_0",
            "active_joint_name": "joint_0",
        },
    }
    recon_cfg={
        "num_initial_pts": 1000,
        "num_kframes": 5,
        "fine_lr": 1e-3
    }
    env = ReconEnv(
        task_cfg=task_cfg,
        explore_cfg=explore_cfg,
        recon_cfg=recon_cfg
    )
    if False:
        from embodied_analogy.representation.obj_repr import Obj_repr
        env.obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
        env.recon_stage(visualize=True)
    else:
        obj_index = task_cfg["obj_cfg"]["asset_path"].split("/")[-1].split("_")[0]
        # env.explore_stage(
        #     save_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore_data.pkl",
        #     visualize=False
        # )
        env.recon_stage(
            load_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore_data.pkl",    
            save_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/recon_data.pkl",
            visualize=False
        )