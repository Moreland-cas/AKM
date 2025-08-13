import sys
import json
import os
import copy
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml

from akm.utility.constants import *

def update_dict(a, b):
    for key, value in b.items():
        if key in a:
            if isinstance(a[key], dict) and isinstance(value, dict):
                update_dict(a[key], value)
            else:
                a[key] = value
        else:
            a[key] = value
            
def filter_tasks(base_yaml_path, yaml_path_list):
    """
    将已经跑过的 task 从 yaml_path_list 中过滤掉
    返回过滤后的 yaml_path_list
    """
    yaml_path_list_filtered = []
    
    for specific_yaml_path in yaml_path_list:
        with open(base_yaml_path, "r") as f:
            base_cfg = yaml.safe_load(f)
            
        with open(specific_yaml_path, "r") as f:
            specific_cfg = yaml.safe_load(f)
                
        task_cfg = copy.deepcopy(base_cfg)
        update_dict(task_cfg, specific_cfg)
        
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
                # 发现三个文件并不同时存在, 因此需要跑
                yaml_path_list_filtered.append(specific_yaml_path)
                continue
        
        if all_exist:
            # 如果 explore 中的 exception 是 CUDA out of memory, 那么还需要跑
            with open(os.path.join(saved_prefix, "explore_result.json"), "r") as f:
                explore_dict = json.load(f)
            if "exception" in explore_dict.keys():
                if "out of memory" in explore_dict["exception"]:
                    yaml_path_list_filtered.append(specific_yaml_path)
                elif "cannot create buffer" in explore_dict["exception"]:
                    yaml_path_list_filtered.append(specific_yaml_path)
                elif "ErrorOutOfHostMemory" in explore_dict["exception"]:
                    yaml_path_list_filtered.append(specific_yaml_path)
                else:
                    continue 
            else:
                print("Skip since all result files already exists.")
                continue 
    return yaml_path_list_filtered
    
        
def test_one(base_yaml_path, specific_yaml_path):
    
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    with open(specific_yaml_path, "r") as f:
        specific_cfg = yaml.safe_load(f)
            
    task_cfg = copy.deepcopy(base_cfg)
    update_dict(task_cfg, specific_cfg)
    method_name = task_cfg["exp_cfg"]["method_name"]
    
    # 否则执行
    if method_name == "ours":
        from akm.simulated_envs.manipulate_env import ManipulateEnv as env_class
    elif method_name == "gflow":
        sys.path.append(PROJECT_ROOT)
        from baselines.generalflow_env import GeneralFlow_ManipEnv as env_class
    elif method_name == "gpnet":
        sys.path.append(PROJECT_ROOT)
        from baselines.gapartnet_env import GAPartNet_ManipEnv as env_class
    
    # for gpu competition
    # import torch
    # useless_var = torch.empty((256, 256, 256, 256), dtype=torch.float32)
    manipulateEnv = env_class(cfg=task_cfg)
    manipulateEnv.main()
    manipulateEnv.delete()
    
    
def test_batch(base_yaml_path, yaml_path_list):
    failed_list = []
    for specific_yaml_path in yaml_path_list:
        try:
            test_one(
                base_yaml_path=base_yaml_path,
                specific_yaml_path=specific_yaml_path,
            )
        except Exception as e:
            print(f"Task {specific_yaml_path} failed: {str(e)}")
            failed_list.append(specific_yaml_path)
    return failed_list
        

def distribute_tasks(tasks, num_groups):
    """将任务列表均匀分配到指定数量的组中"""
    # 首先对于 tasks 这个 list 进行 sort
    tasks.sort()
    
    # 计算每组的基本大小和余数
    base_size = len(tasks) // num_groups
    remainder = len(tasks) % num_groups
    
    distributed = []
    start = 0
    
    for i in range(num_groups):
        # 前remainder组多分配一个任务
        end = start + base_size + (1 if i < remainder else 0)
        distributed.append(tasks[start:end])
        start = end
    
    return distributed

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda_idx', type=str)
    parser.add_argument('--ts', help="total split, split the tasks into e.g. 4 split", type=int, default=4)
    parser.add_argument('--cs', help="current split, e.g. one of [0, 1, 2, 3] when total split is 4", type=int, default=0)
    
    # byp for base yaml path
    parser.add_argument('--byp', type=str, default="gflow_draw")
    parser.add_argument('--task_cfgs_folder', type=str, default="/home/zby/Programs/AKM/cfgs/task_cfgs_new")
    
    args = parser.parse_args()
    assert (args.cs >= 0) and (args.cs < args.ts)
    
    yaml_path_list = []
    for yaml_path in os.listdir(args.task_cfgs_folder): 
        yaml_path_list.append(os.path.join(args.task_cfgs_folder, yaml_path))
    
    # 可视化五个 revolute, 五个 prismatic, 最好都是一次 explore 就成功的那种, 且物品最好不同
    yaml_path_list=[
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/11.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/17.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/22.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/27.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/33.yaml",
        #
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/85.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/93.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/104.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/110.yaml",
        "/home/zby/Programs/AKM/cfgs/task_cfgs_new/115.yaml",
    ]
    
    # 对 yaml_path_list 进行过滤
    base_yaml_path = os.path.join("/home/zby/Programs/AKM/cfgs/", f"{args.byp}.yaml")
    yaml_path_list = filter_tasks(base_yaml_path, yaml_path_list)
    
    task_groups = distribute_tasks(yaml_path_list, args.ts)
    current_group = task_groups[args.cs]
    
    failed_list = test_batch(
        base_yaml_path=base_yaml_path,
        yaml_path_list=current_group,
    )
    if len(failed_list) == 0:
        print("All done!")
    else:
        print(f"There are {len(failed_list)} failed case:")
        for failed_case in failed_list:
            print(failed_case)
    