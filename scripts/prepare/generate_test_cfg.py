import os
import json
import argparse
from embodied_analogy.utility.constants import ASSET_PATH, PROJECT_ROOT
from embodied_analogy.utility.randomize_obj_pose import randomize_obj

if __name__ == "__main__":
    cfgs = {}

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
        cfgs.update(tmp_dict)

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
        cfgs.update(tmp_dict)

    # with open(os.path.join(PROJECT_ROOT, "scripts", "test_data.json"), 'w', encoding='utf-8') as f:
    #     json.dump(cfg, f, ensure_ascii=False, indent=4)

    # total_list = [(k, v) for k,v in cfg.items()]
    # import random
    # random.shuffle(total_list)
    # cfg = {k:v for k,v in total_list[:10]}
    # with open(os.path.join(PROJECT_ROOT, "scripts", "test_data_subset.json"), 'w', encoding='utf-8') as f:
    #     json.dump(cfg, f, ensure_ascii=False, indent=4)    

    # 这里得到了完整的 cfg_dict, 现在需要将其以一个文件结构保存下来
    parser = argparse.ArgumentParser(description='Folder to save the cfg files')
    parser.add_argument('--log_dir', type=str, default="/home/zby/Programs/Embodied_Analogy/assets/logs", help='root folder to save all the runs')
    parser.add_argument('--run_name', type=str, help='name of the run folder')
    parser.add_argument('--load_obj_seed', type=int, default=666, help='Random seed of loading obj with 0 joint state')
    args = parser.parse_args()
    
    for tmp_folder, cfg in cfgs.items():
        # 在这里对于位姿进行添加扰动
        cfg = randomize_obj(cfg)
        
        # 在这里写入 instruction 和 init_joint_state
        cfg["instruction"] = "open the " + cfg["obj_description"]
        cfg["init_joint_state"] = 0.0
        
        cfg_folder_path = os.path.join(args.log_dir, args.run_name, tmp_folder)
        os.makedirs(cfg_folder_path, exist_ok=True)
        
        cfg_file_path = os.path.join(cfg_folder_path, "cfg.json")
        with open(cfg_file_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
            
    print("Generate cfg files done.")
