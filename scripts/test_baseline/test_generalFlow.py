# make sure to `conda activate general_flow` before sun this script
from embodied_analogy.utility.constants import PROJECT_ROOT
import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.join(PROJECT_ROOT, "baselines/GeneralFlow"))
from ea_baseline import GeneralFlow_ManipEnv

# 遍历所有 cfg, 然后跑实验
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help='Manip run name')
parser.add_argument('--save_run_name', type=str, help='saved general flow Manip run name')
args = parser.parse_args()

import json
root_path = os.path.join(PROJECT_ROOT, "assets/logs", args.run_name)
for object_folder in os.listdir(root_path):
    # /logs/manip_4_16/45135_1_prismatic
    object_path = os.path.join(root_path, object_folder)
    manip_types = ["close", "open"]
    for manip_type in manip_types:
        # /logs/manip_4_16/45135_1_prismatic/open
        manip_folder = os.path.join(object_path, manip_type)
        # 遍历不同 scale 的文件夹
        for scale_folder in os.listdir(manip_folder):
            # /logs/manip_4_16/45135_1_prismatic/open/scale_1
            scale_path = os.path.join(manip_folder, scale_folder)
            save_folder = scale_path.replace(args.run_name, args.save_run_name)
            os.makedirs(save_folder, exist_ok=True)
            
            original_stdout = sys.stdout
            output_txt = os.path.join(save_folder, "output.txt")
            with open(output_txt, "w") as f:
                sys.stdout = f
                
                # 如果 save_folder 底下已经有了 result.pkl, 则跳过
                if os.path.isfile(os.path.join(save_folder, "result.pkl")):
                    # sys.stdout = original_stdout
                    print(f"result.pkl already exists in {save_folder}")
                    continue
                
                # 因为有些文件夹（没有成功 explore 的那些）没有 cfg.json，所以需要try
                try:
                    with open(os.path.join(scale_path, "cfg.json"), 'r', encoding='utf-8') as file:
                        manip_cfg = json.load(file)
                        # joint_type = manip_cfg["joint_type"]
                        
                except Exception as e:
                    print(f"Skip {scale_path}: {e}")
                    continue
                
                # Test general flow baseline
                env = GeneralFlow_ManipEnv(cfg=manip_cfg)
                manip_result = env.main_general_flow(visualize=False, gt_joint_type=True)
                print(manip_result)
                
                with open(os.path.join(save_folder, 'result.pkl'), 'wb') as f:
                    pickle.dump(manip_result, f)
            sys.stdout = original_stdout
            
print("Test general flow baseline, done!")
