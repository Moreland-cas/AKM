import os
import json
from embodied_analogy.utility.constants import ASSET_PATH, PROJECT_ROOT
cfg = {}

pri_path = os.path.join(ASSET_PATH, "dataset/one_drawer_cabinet")
for tmp_folder in os.listdir(pri_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        tmp_folder: {
            "joint_type": "prismatic",
            "data_path": os.path.join("dataset/one_drawer_cabinet", tmp_folder),
            "obj_index": obj_index,
            "joint_index": joint_index,
            "obj_description": "cabinet",
            "load_pose": None,
            "load_quat": None,
            "load_scale": None,
            "active_link_name": f"link_{joint_index}",
            "active_joint_name": f"joint_{joint_index}"
        }
    }
    cfg.update(tmp_dict)

rev_path = os.path.join(ASSET_PATH, "dataset/one_door_cabinet")
for tmp_folder in os.listdir(rev_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        tmp_folder: {
            "joint_type": "revolute",
            "data_path": os.path.join("dataset/one_door_cabinet", tmp_folder),
            "obj_index": obj_index,
            "joint_index": joint_index,
            "obj_description": "cabinet",
            "load_pose": None,
            "load_quat": None,
            "load_scale": None,
            "active_link_name": f"link_{joint_index}",
            "active_joint_name": f"joint_{joint_index}"
        }
    }
    cfg.update(tmp_dict)

with open(os.path.join(PROJECT_ROOT, "scripts", "test_data.json"), 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)

total_list = [(k, v) for k,v in cfg.items()]
import random
random.shuffle(total_list)
cfg = {k:v for k,v in total_list[:10]}
with open(os.path.join(PROJECT_ROOT, "scripts", "test_data_subset.json"), 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)    
    