import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml

from embodied_analogy.environment.manipulate_env import ManipulateEnv
from embodied_analogy.utility.constants import *


def test(test_cfgs):
    for idx, test_cfg in test_cfgs.items():
        test_cfg.update({
            
        })
        manipulateEnv = ManipulateEnv(cfg=test_cfg)
        overall_result = manipulateEnv.main()
        manipulateEnv.delete()
        save_path = ""
        break


cfg = {
    "joint_type": "prismatic",
    "data_path": "dataset/one_drawer_cabinet/41083_link_3",
    "obj_index": "41083",
    "joint_index": "3",
    "obj_description": "cabinet",
    "load_pose": [
        0.8528176546096802,
        0.0,
        0.5067352652549744
    ],
    "load_quat": [
        0.9986352920532227,
        0.01783677004277706,
        -0.0008765950915403664,
        -0.0490783266723156
    ],
    "load_scale": 1,
    "active_link_name": "link_3",
    "active_joint_name": "joint_3",
    "instruction": "open the cabinet",
    "init_joint_state": 0.24008742437156694,
    "obj_folder_path_explore": "/home/zby/Programs/Embodied_Analogy/assets/logs/explore_512/41083_link_3",
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "fully_zeroshot": False,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "valid_thresh": 0.5,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "num_initial_pts": 1000,
    "offscreen": True,
    "use_anygrasp": True,
    "obj_folder_path_reconstruct": "/home/zby/Programs/Embodied_Analogy/assets/logs/recon_512/41083_link_3",
    "num_kframes": 5,
    "fine_lr": 0.001,
    "save_memory": True,
    "scale_dir": "/home/zby/Programs/Embodied_Analogy/assets/logs/manip_512/41083_link_3/close/scale_0.1",
    "manipulate_type": "close",
    "manipulate_distance": 0.1,
    "reloc_lr": 0.003,
    "whole_traj_close_loop": True,
    "max_manip": 5.0,
    "prismatic_whole_traj_success_thresh": 0.01,
    "revolute_whole_traj_success_thresh": 5.0,
    "max_attempts": 5,
    "max_distance": 0.3,
    "prismatic_reloc_interval": 0.05,
    "prismatic_reloc_tolerance": 0.01,
    "revolute_reloc_interval": 5.0
}

with open("/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg)
manipulateEnv = ManipulateEnv(cfg=cfg)
overall_result = manipulateEnv.main()
manipulateEnv.delete()