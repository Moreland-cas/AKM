import os
import sys
import json
import copy
import yaml
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# avoid huggingface from online connection
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


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
    Filters out already-running tasks from the yaml_path_list. 
    Returns the filtered yaml_path_list.
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
                # Found that the three files do not exist at the same time, so you need to run
                yaml_path_list_filtered.append(specific_yaml_path)
                continue
        
        if all_exist:
            # If the exception in explore is CUDA out of memory, you also need to run
            with open(os.path.join(saved_prefix, "explore_result.json"), "r") as f:
                explore_dict = json.load(f)
            if "exception" in explore_dict.keys():
                if "Remote end closed connection without response" in explore_dict["exception"]:
                    yaml_path_list_filtered.append(specific_yaml_path)
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
    
    if method_name == "ours":
        from akm.simulated_envs.manipulate_env import ManipulateEnv as env_class
    elif method_name == "gflow":
        sys.path.append(PROJECT_ROOT)
        from baselines.generalflow_env import GeneralFlow_ManipEnv as env_class
    elif method_name == "gpnet":
        sys.path.append(PROJECT_ROOT)
        from baselines.gapartnet_env import GAPartNet_ManipEnv as env_class
    
    manipulateEnv = env_class(cfg=task_cfg)
    manipulateEnv.main()
    manipulateEnv.delete()
    
    
def test_batch(base_yaml_path, yaml_path_list):
    failed_list = []
    for specific_yaml_path in yaml_path_list:
        try:
        # if True:
            test_one(
                base_yaml_path=base_yaml_path,
                specific_yaml_path=specific_yaml_path,
            )
        except Exception as e:
            print(f"Task {specific_yaml_path} failed: {str(e)}")
            failed_list.append(specific_yaml_path)
    return failed_list
        

def distribute_tasks(tasks, num_groups):
    """
    Evenly distribute the task list into the specified number of groups
    """
    tasks.sort()
    
    base_size = len(tasks) // num_groups
    remainder = len(tasks) % num_groups
    
    distributed = []
    start = 0
    
    for i in range(num_groups):
        end = start + base_size + (1 if i < remainder else 0)
        distributed.append(tasks[start:end])
        start = end
    
    return distributed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts', help="total split, split the tasks into e.g. 4 split", type=int, default=40)
    parser.add_argument('--cs', help="current split, e.g. one of [0, 1, 2, 3] when total split is 4", type=int, default=1)
    
    # method_cfg for base yaml path
    parser.add_argument('--method_cfg', type=str, default="ours")
    parser.add_argument('--task_cfgs_folder', type=str, default=os.path.join(PROJECT_ROOT, "cfgs/simulation_cfgs/tasks"))
    
    args = parser.parse_args()
    assert (args.cs >= 0) and (args.cs < args.ts)
    
    yaml_path_list = []
    for yaml_path in os.listdir(args.task_cfgs_folder): 
        yaml_path_list.append(os.path.join(args.task_cfgs_folder, yaml_path))
    
    # filter yaml_path_list
    base_yaml_path = os.path.join(PROJECT_ROOT, "cfgs/simulation_cfgs/methods", f"{args.method_cfg}.yaml")
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
    