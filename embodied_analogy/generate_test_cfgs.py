import os
import copy
import json
import numpy as np

data_root = "/home/zby/Programs/Embodied_Analogy/assets/dataset/"
prismatic_data_root = "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_drawer_cabinet"
revolute_data_root = "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet"

prismatic_dirs = os.listdir(prismatic_data_root)
revolute_dirs = os.listdir(revolute_data_root)


base_cfg = {
    "base_cfg" :{
        "phy_timestep": 1/250.,
        "planner_timestep": None,
        "use_sapien2": True 
    },
    "robot_cfg" :{},
    "explore_cfg" :{
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "max_tries": 100,
        "update_sigma": 0.05
    },
    "recon_cfg" :{
        "num_initial_pts": 1000,
        "num_kframes": 5,
        "fine_lr": 1e-3
    },
    "manip_cfg" :{
        "reloc_lr": 3e-3,
        "reserved_distance": 0.05
    },
    "task_cfg" :{
        "instruction": None,
        "obj_description": None,
        "delta": None,
        "obj_cfg": {
            "asset_path": None, 
            "scale": 1.,
            "active_link_name": None,
            "active_joint_name": None,
        }
    }
}

"""
/logs
    /prismatic
        /open
            /0.05
            /0.1
            /0.2
        /close
            /0.05
            /0.1
            /0.2
    /revolute
        /open
            /0.05
            /0.1
            /0.2
        /close
            /0.05
            /0.1
            /0.2
"""
dump_dict = {
    "prismatic": {
        "open": {},
        "close": {}
    }
}

prismatic_delta = [0.05, 0.1, 0.2]
for prismatic_dir in prismatic_dirs:
    tmp_cfg = copy.deepcopy(base_cfg)
    # 从 prismatic_dir 中解析出必要的信息
    tmp_cfg["task_cfg"]["instruction"] = "open the cabinet"
    tmp_cfg["task_cfg"]["obj_description"] = "cabinet"
    tmp_cfg["task_cfg"]["delta"] = 0.2
    
    active_link_name = "link_" + prismatic_dir.split("_")[-1]
    active_joint_name = "joint_" + prismatic_dir.split("_")[-1]
    tmp_cfg["task_cfg"]["obj_cfg"]["active_link_name"] = active_link_name
    tmp_cfg["task_cfg"]["obj_cfg"]["active_joint_name"] = active_joint_name
    tmp_cfg["task_cfg"]["obj_cfg"]["asset_path"] = os.path.join(prismatic_data_root, prismatic_dir)
    path = "/home/zby/Programs/Embodied_Analogy/logs/prismatic_logs"
    dump_dict["prismatic_test_cfgs"].append(tmp_cfg)


dump_dict["revolute_test_cfgs"] = []
for revolute_dir in revolute_dirs:
    tmp_cfg = copy.deepcopy(base_cfg)
    # 从 prismatic_dir 中解析出必要的信息
    tmp_cfg["task_cfg"]["instruction"] = "open the cabinet"
    tmp_cfg["task_cfg"]["obj_description"] = "cabinet"
    tmp_cfg["task_cfg"]["delta"] = np.deg2rad(30)
    
    active_link_name = "link_" + prismatic_dir.split("_")[-1]
    active_joint_name = "joint_" + prismatic_dir.split("_")[-1]
    tmp_cfg["task_cfg"]["obj_cfg"]["active_link_name"] = active_link_name
    tmp_cfg["task_cfg"]["obj_cfg"]["active_joint_name"] = active_joint_name
    tmp_cfg["task_cfg"]["obj_cfg"]["asset_path"] = os.path.join(revolute_data_root, revolute_dir)
    dump_dict["revolute_test_cfgs"].append(tmp_cfg)


# 将字典保存为 JSON 文件
with open('./test.json', 'w', encoding='utf-8') as json_file:
    json.dump(dump_dict, json_file, ensure_ascii=False, indent=4)
    