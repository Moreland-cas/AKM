import math
import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    image_to_camera,
    visualize_pc,
    get_depth_mask
)
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
                "fine_lr": 1e-3
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
    
    def reconstruct(self, visualize=False):
        self.obj_repr.reconstruct(
            num_initial_pts=self.num_initial_pts,
            num_kframes=self.num_kframes,
            obj_description=self.obj_description,
            fine_lr=self.fine_lr,
            file_path=None,
            gt_joint_dir_w=None,
            visualize=visualize,
        )
        # TODO: 这里加一个返回是否有 error 的判断
        
    def recon_main(self, visualize=False):
        
        self.explore_main(visualize=visualize)
        
        self.reconstruct(visualize=visualize)
        
    
if __name__ == "__main__":
    # obj_idx = 7221
    obj_idx = 44962
    obj_repr_path = f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/explore/explore_data.pkl"
    obj_repr_data = Obj_repr.load(obj_repr_path)
    # obj_repr_data.frames.frame_list.reverse()
    
    obj_repr_data.reconstruct(
        num_initial_pts=1000,
        num_kframes=5,
        visualize=False,
        gt_joint_dir_w=np.array([-1, 0, 0]),
        # gt_joint_dir_w=np.array([0, 0, 1]),
        # gt_joint_dir_w=None,
        file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/reconstruct/recon_data.pkl"
        # file_path = None
    )
    