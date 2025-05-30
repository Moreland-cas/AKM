import json
import copy
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml

from embodied_analogy.environment.manipulate_env import ManipulateEnv
from embodied_analogy.utility.constants import *


def update_dict(a, b):
    for key, value in b.items():
        if key in a:
            if isinstance(a[key], dict) and isinstance(value, dict):
                update_dict(a[key], value)
            else:
                a[key] = value
        else:
            a[key] = value
            
            
def test_all(base_yaml_path, test_yaml_folder):
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    
    yaml_paths = os.listdir(test_yaml_folder)
    for yaml_path in yaml_paths:
        with open(os.path.join(test_yaml_folder, yaml_path), "r") as f:
            specific_cfg = yaml.safe_load(f)
            
        task_cfg = copy.deepcopy(base_cfg)
        update_dict(task_cfg, specific_cfg)
        
        # temporary test
        if task_cfg["task_cfg"]["task_id"] != 60:
            continue
              
        
        # 首先检查该路径下有没有运行完成标志文件, 如果有则 continue
        saved_prefix = os.path.join(task_cfg["exp_cfg"]["exp_folder"], str(task_cfg["task_cfg"]["task_id"]))
        result_file_names = [
            "explore_result.json",
            "recon_result.json",
            "manip_result.json"
        ]
        saved_result_paths = [os.path.join(saved_prefix, result_file_name) for result_file_name in result_file_names]
        all_exist = True
        for saved_result_path in saved_result_paths:
            if not os.path.exists(saved_result_path):
                all_exist = False
                break
        
        if all_exist:
            print("Skip since all result files already exists.")
            continue
        
        # 否则执行
        manipulateEnv = ManipulateEnv(cfg=task_cfg)
        manipulateEnv.main()
        manipulateEnv.delete()
        # 并且产生运行完成标志文件


if __name__ == "__main__":
    base_yaml_path = "/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml"
    test_yaml_folder = "/home/zby/Programs/Embodied_Analogy/scripts/task_cfgs"
    test_all(
        base_yaml_path=base_yaml_path,
        test_yaml_folder=test_yaml_folder
    )
    