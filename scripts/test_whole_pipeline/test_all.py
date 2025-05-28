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
        
        manipulateEnv = ManipulateEnv(cfg=task_cfg)
        manipulateEnv.main()
        manipulateEnv.delete()
        break


if __name__ == "__main__":
    base_yaml_path = "/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml"
    test_yaml_folder = "/home/zby/Programs/Embodied_Analogy/scripts/task_cfgs"
    test_all(
        base_yaml_path=base_yaml_path,
        test_yaml_folder=test_yaml_folder
    )
    