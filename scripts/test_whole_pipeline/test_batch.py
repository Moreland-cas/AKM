import sys
import json
import os
import copy
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml

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
            
            
def test_one(base_yaml_path, specific_yaml_path, method_name):
    
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    with open(specific_yaml_path, "r") as f:
        specific_cfg = yaml.safe_load(f)
            
    task_cfg = copy.deepcopy(base_cfg)
    update_dict(task_cfg, specific_cfg)
    
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
        # 如果 explore 中的 exception 是 CUDA out of memory, 那么还需要跑
        with open(os.path.join(saved_prefix, "explore_result.json"), "r") as f:
            explore_dict = json.load(f)
        if "exception" in explore_dict.keys():
            if not explore_dict["exception"].startswith("CUDA out of memory."):
                return 
        else:
            print("Skip since all result files already exists.")
            return 
    
    # 否则执行
    if method_name == "ours":
        from embodied_analogy.environment.manipulate_env import ManipulateEnv as env_class
    elif method_name == "gflow":
        sys.path.append(PROJECT_ROOT)
        from baselines.generalflow_env import GeneralFlow_ManipEnv as env_class
    elif method_name == "gpnet":
        sys.path.append(PROJECT_ROOT)
        from baselines.gapartnet_env import GAPartNet_ManipEnv as env_class
        
    manipulateEnv = env_class(cfg=task_cfg)
    manipulateEnv.main()
    manipulateEnv.delete()
    
    
def test_batch(base_yaml_path, yaml_path_list, method_name="ours"):
    failed_list = []
    for specific_yaml_path in yaml_path_list:
        try:
            test_one(
                base_yaml_path=base_yaml_path,
                specific_yaml_path=specific_yaml_path,
                method_name=method_name
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
    
    parser.add_argument('--method', type=str, default="ours") # ours, gpnet, gflow
    parser.add_argument('--base_yaml_path', type=str, default="/home/zby/Programs/Embodied_Analogy/cfgs/base_6_39.yaml")
    parser.add_argument('--task_cfgs_folder', type=str, default="/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new")
    
    args = parser.parse_args()
    yaml_path_list = []
    for yaml_path in os.listdir(args.task_cfgs_folder): 
        yaml_path_list.append(os.path.join(args.task_cfgs_folder, yaml_path))
    
    yaml_path_list=[
        # "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/3.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/4.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/7.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/10.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/78.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/95.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/97.yaml",
        "/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs_new/114.yaml",
    ]
    
    task_groups = distribute_tasks(yaml_path_list, args.ts)
    
    assert (args.cs >= 0) and (args.cs < args.ts)
    current_group = task_groups[args.cs]
    
    failed_list = test_batch(
        base_yaml_path=args.base_yaml_path,
        yaml_path_list=current_group,
        method_name=args.method
    )
    if len(failed_list) == 0:
        print("All done!")
    else:
        print(f"There are {len(failed_list)} failed case:")
        for failed_case in failed_list:
            print(failed_case)
    