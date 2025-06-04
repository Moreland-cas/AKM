
import json
import subprocess
import os
import multiprocessing
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
    
    
def worker(gpu_id, base_yaml_path, yaml_path_list):
    """处理单个GPU所有任务的函数"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['VK_ICD_FILENAMES'] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    os.environ['DISPLAY'] = ""
    print(f"GPU {gpu_id} start process {len(yaml_path_list)} tasks.")
    
    for specific_yaml_path in yaml_path_list:
        try:
            test_one(
                base_yaml_path=base_yaml_path,
                specific_yaml_path=specific_yaml_path
            )
        except Exception as e:
            print(f"任务 {specific_yaml_path} 执行失败: {str(e)}")
        
        break


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


def use_multi_process():
    base_yaml_path = "/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml"
    test_yaml_folder = "/home/zby/Programs/Embodied_Analogy/scripts/task_cfgs"
    
    # 得到所有任务的参数列表
    yaml_path_list = []
    for yaml_path in os.listdir(test_yaml_folder): 
        yaml_path_list.append(os.path.join(test_yaml_folder, yaml_path))
    
    usable_gpu_ids = [3, 4, 5, 6]
    # usable_gpu_ids = [3]
    task_groups = distribute_tasks(yaml_path_list, len(usable_gpu_ids))
    
    task_distribution = dict(zip(usable_gpu_ids, task_groups))
    # 创建并启动进程
    processes = []
    for gpu_id, yaml_paths in task_distribution.items():
        p = multiprocessing.Process(
            target=worker,
            args=(gpu_id, base_yaml_path, yaml_paths)
        )
        # 设置为守护进程, 这样在父进程退出后子进程也会退出
        p.daemon = True
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("All done!")

def worker_subprocess(gpu_id, inputs):
    """处理单个GPU所有任务的函数"""
    for input_val in inputs:
        cmd = [
            "python",
            "test.py",
            "--gpu", str(gpu_id),
            "--input", str(input_val)
        ]
        print(f"GPU {gpu_id} executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
def use_subprocess():
    base_yaml_path = "/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml"
    test_yaml_folder = "/home/zby/Programs/Embodied_Analogy/scripts/task_cfgs"
    
    # 得到所有任务的参数列表
    yaml_path_list = []
    for yaml_path in os.listdir(test_yaml_folder): 
        yaml_path_list.append(os.path.join(test_yaml_folder, yaml_path))
    
    # usable_gpu_ids = [1, 2, 3, 4, 5, 6, 7]
    usable_gpu_ids = [1, 2]
    # usable_gpu_ids = [3]
    task_groups = distribute_tasks(yaml_path_list, len(usable_gpu_ids))
    
    task_distribution = dict(zip(usable_gpu_ids, task_groups))
    # 将这个文件进行保存
    tmp_save_folder = "/home/zby/Programs/Embodied_Analogy/scripts/test_whole_pipeline/batch_yamls"
    os.makedirs(tmp_save_folder, exist_ok=True)
    
    for k, v in task_distribution.items():
        with open(os.path.join(tmp_save_folder, f"gpu_{k}.yaml"), "w") as f:
            json.dump(v, f)
        
    # 创建并启动进程
    processes = []
    for gpu_id, yaml_paths in task_distribution.items():
        batch_yaml_path = os.path.join(tmp_save_folder, f"gpu_{gpu_id}.yaml")
        # 构造命令列表
        command = [
            "python", "/home/zby/Programs/Embodied_Analogy/scripts/test_whole_pipeline/test_batch.py",
            "--cuda_idx", str(gpu_id),
            "--base_yaml_path", base_yaml_path,
            "--batch_yaml_path", batch_yaml_path
        ]

        # 启动子进程
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)

        # stdout, stderr = process.communicate()

        # # 打印输出和错误
        # print("标准输出：", stdout)
        # print("标准错误：", stderr)

    for process in processes:
        process.wait()
        
    print("All done!")
        
if __name__ == "__main__":
    # use_multi_process()
    use_subprocess()