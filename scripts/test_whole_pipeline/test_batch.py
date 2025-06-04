import json
import os
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
            
            
def test_one(base_yaml_path, specific_yaml_path):
    
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
        print("Skip since all result files already exists.")
        return 
    
    # 否则执行
    manipulateEnv = ManipulateEnv(cfg=task_cfg)
    manipulateEnv.main()
    manipulateEnv.delete()
    
    
def test_batch(base_yaml_path, yaml_path_list):
    # 这里直接跑个 10 遍, 由于我们的程序会检测是否已经跑完特定任务, 所以不用担心重复跑, 跑 10 遍可以有效降低系统问题导致的程序崩溃
    # for i in range(10):
    for specific_yaml_path in yaml_path_list:
        try:
            test_one(
                base_yaml_path=base_yaml_path,
                specific_yaml_path=specific_yaml_path
            )
        except Exception as e:
            print(f"Task {specific_yaml_path} failed: {str(e)}")
        

def distribute_tasks(tasks, num_groups):
    """将任务列表均匀分配到指定数量的组中"""
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda_idx', type=str)
    parser.add_argument('--total_split', help="Split the tasks into #total_split splits, e.g. 4 split", type=int, default=4)
    parser.add_argument('--current_split', help="run #current_split of total splits, e.g. one of [0, 1, 2, 3] when total split is 4", type=int, default=1)
    
    parser.add_argument('--base_yaml_path', type=str, default="/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml")
    parser.add_argument('--specific_yaml_folder', type=str, default="/home/zby/Programs/Embodied_Analogy/cfgs/task_cfgs")
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_idx
    
    yaml_path_list = []
    for yaml_path in os.listdir(args.specific_yaml_folder): 
        yaml_path_list.append(os.path.join(args.specific_yaml_folder, yaml_path))
        
    task_groups = distribute_tasks(yaml_path_list, args.total_split)
    
    assert (args.current_split >= 0) and (args.current_split < args.total_split)
    current_group = task_groups[args.current_split]
    
    test_batch(
        base_yaml_path=args.base_yaml_path,
        yaml_path_list=current_group
    )
    print("All done")
    