import os
import json

cfg = {}

asset_path = "/media/zby/MyBook/embody_analogy_data/assets"
pri_path = os.path.join(asset_path, "dataset/one_drawer_cabinet")
for tmp_folder in os.listdir(pri_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        tmp_folder: {
            "joint_type": "prismatic",
            "asset_path": os.path.join(pri_path, tmp_folder),
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

rev_path = os.path.join(asset_path, "dataset/one_door_cabinet")
for tmp_folder in os.listdir(rev_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        tmp_folder: {
            "joint_type": "revolute",
            "asset_path": os.path.join(rev_path, tmp_folder),
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

use_subset = False
if use_subset:
    total_list = [(k, v) for k,v in cfg.items()]
    import random
    random.shuffle(total_list)
    cfg = {k:v for k,v in total_list[:10]}
    with open(os.path.join("/home/zby/Programs/Embodied_Analogy/scripts", "test_data_subset.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
else:
    with open(os.path.join("/home/zby/Programs/Embodied_Analogy/scripts", "test_data.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    