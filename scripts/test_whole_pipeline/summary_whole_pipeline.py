"""
给定一个 yaml 的路径, 根据这个 yaml 找到对应的文件夹, 读取里面的 k 个任务
"""
import os
import json
import yaml
import numpy as np
from embodied_analogy.utility.utils import (
    MinKNumbers
)

############### 读取数据 ###############
task_yaml_path = "/home/zby/Programs/Embodied_Analogy/cfgs/base.yaml"
with open(task_yaml_path, "r") as f:
    task_yaml = yaml.safe_load(f)
# "/home/zby/Programs/Embodied_Analogy/assets/logs/test"
log_folder = task_yaml["exp_cfg"]["exp_folder"]

saved_result = {}
run_folders = os.listdir(log_folder)
for run_folder in run_folders:
    run_path = os.path.join(log_folder, run_folder)
    
    try: # TODO 之后把 try-except 去掉
        saved_result[int(run_folder)] = {}
        
        with open(os.path.join(run_path, "explore_result.json"), "r") as f:
            explore_result = json.load(f)
            saved_result[int(run_folder)]["explore"] = explore_result
        
        with open(os.path.join(run_path, "recon_result.json"), "r") as f:
            recon_result = json.load(f)
            saved_result[int(run_folder)]["recon"] = recon_result
        
        with open(os.path.join(run_path, "manip_result.json"), "r") as f:
            manip_result = json.load(f)
            saved_result[int(run_folder)]["manip"] = manip_result
    
    except Exception as e:
        pass

# 这里要确保每个 run 的结果都读取进来了
assert len(saved_result.keys()) == len(run_folders)

############### summary_explore ###############
# 直接把一些判断是否成功的超参放到这里, 方便修改
EXPLORE_PRISMATIC_VALID = 0.05
EXPLORE_REVOLUTE_VALID = 5

def explore_actually_valid(result):
    joint_type = result["joint_type"]
    joint_delta = result["joint_state_end"] - result["joint_state_start"]
    
    if joint_type == "prismatic":
        return joint_delta >= EXPLORE_PRISMATIC_VALID
    
    elif joint_type == "revolute":
        return np.rad2deg(joint_delta) >= EXPLORE_REVOLUTE_VALID

# 写的时候直接把函数的输入改为 saved_result 这个 dict, 方便假如我们只想跑 explore 以及 explore 的 summary
def summary_explore(saved_result, task_yaml):
    max_tries = task_yaml["explore_env_cfg"]["max_tries"]

    # 到此为止把所有有 result 的读取进来了
    num_total_exp = len(saved_result.keys()) # 所有的实验
    num_total_tries = 0

    # 记录下 tries 从 1 到 MAX_TRIES 个的 pred_valid 个数和 actual_valid 个数
    num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
    num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}

    # 用于记录实际的状态变化最小的 k 个, 用于确定 explore valid 的阈值
    prismatic_heap = MinKNumbers()
    revolute_heap = MinKNumbers()

    # 转换为之前代码可复用的格式
    # TODO 这里到时候记得改回去
    # results = [saved_result[k]["explore"] for k in saved_result.keys()]
    results = [saved_result[k]["explore"] for k in saved_result.keys() if "explore" in saved_result[k].keys()]
    for i, result in enumerate(results):
        num_tries = result["num_tries"]
        num_total_tries += num_tries
        has_valid_explore = result["has_valid_explore"]
        
        if has_valid_explore:
            joint_delta = result["joint_state_end"] - result["joint_state_start"]
            if result["joint_type"] == "prismatic":
                prismatic_heap.add_number(abs(joint_delta))
            elif result["joint_type"] == "revolute":
                revolute_heap.add_number(np.rad2deg(abs(joint_delta)))
        
        for i in range(num_tries, max_tries + 1):
            if has_valid_explore:
                num_pred_valid[i] += 1
            
                if explore_actually_valid(result):
                    num_actual_valid[i] += 1
                    
    # 打印结果
    print(f"Total successful (actual): {num_actual_valid[max_tries]} / {num_total_exp} = {num_actual_valid[max_tries] / num_total_exp:.2f}")
    print(f"Total successful (pred): {num_pred_valid[max_tries]} / {num_total_exp} = {num_pred_valid[max_tries] / num_total_exp:.2f}")
    print(f"Average number of triess: {num_total_tries} / {num_total_exp} = {num_total_tries / num_total_exp:.2f}")

    print("jonit delta statistic (for pred as valid):")
    print("prismatic:", prismatic_heap.get_min_k(10))
    print("revolute:", revolute_heap.get_min_k(10))

    # 打印每个尝试次数的成功率
    print("Detailed Results:")
    for tries in range(1, max_tries + 1):
        success_rate_pred = num_pred_valid[tries] / num_total_exp
        success_rate_actual = num_actual_valid[tries] / num_total_exp
        print(f"tries={tries} | actual success rate: {success_rate_actual:.2f} | pred success rate {success_rate_pred:.2f}")


############### summary_reconstruct ###############
def print_recon_result(saved_result, prismatic_thresh, revolute_thresh):
    """
    统计误差在 prismatic_thresh (m) 和 revolute_thresh (degree) 范围内的重建数量, 以及定量误差
   
    分析一下重建失败的原因, 有一些是因为根本没有有效的 explore, 有一些是因为我认为有有效的 explore, 其实没有
    那剩下的就是 explore 的尺度挺大的, 但还是失败了, 这些之中又分为 type 估计错误, 最后仅对 type_loss 为 0 的计算定量误差 
    (NOTE 因为那些 type_loss 不为 0 的定量 loss 是没有意义的)
    """
    explore_results = []
    recon_results = []
    for k, v in saved_result.items():
        # TODO 这里的 "explore" 和 "recon" 都应该有的
        if ("recon" in v.keys()) and ("explore" in v.keys()):
            explore_results.append(v["explore"])
            recon_results.append(v["recon"])
            
    # num_exp = len(saved_result.keys())
    # TODO 记得改回来
    num_exp = len(explore_results)
    num_prismatic = 0
    num_revolute = 0
    
    # 先根据 "什么样的 reconstruct 是好的", 计算整体的成功率, 以及成功的那些的误差
    num_prismatic_success = 0
    num_revolute_success = 0
    
    prismatic_angle_err_coarse = []
    revolute_angle_err_coarse = []
    revolute_pos_err_coarse = []
    
    prismatic_angle_err_fine = []
    revolute_angle_err_fine = []
    revolute_pos_err_fine = []
    
    # 统计所有的任务, 根据 recon_success_thresh 计算 success_rate
    for i, recon_result in enumerate(recon_results):
        # 在这里判断一下是否 reconstruct 为 True
        if not recon_result["has_valid_reconstruct"]:
            joint_type = explore_results[i]["joint_type"]
            if joint_type == "prismatic":
                num_prismatic += 1
            else:
                num_revolute += 1
            continue
        
        joint_type = recon_result["gt_w"]["joint_type"]
        coarse_loss = recon_result['coarse_loss']
        fine_loss = recon_result['fine_loss']
        
        coarse_type_loss = coarse_loss['type_err']
        coarse_angle_err = coarse_loss['angle_err']
        coarse_pos_err = coarse_loss['pos_err']
        
        fine_type_loss = fine_loss['type_err']
        fine_angle_err = fine_loss['angle_err']
        fine_pos_err = fine_loss['pos_err']
        
        assert (coarse_type_loss == fine_type_loss) and "coarse_type_loss and fine_type_loss should be the same"
        
        if joint_type == "prismatic":
            num_prismatic += 1
            if fine_angle_err < np.deg2rad(revolute_thresh) and fine_type_loss == 0:
                num_prismatic_success += 1
                prismatic_angle_err_coarse.append(coarse_angle_err)
                prismatic_angle_err_fine.append(fine_angle_err)
        else:
            num_revolute += 1
            if fine_pos_err < prismatic_thresh and fine_angle_err < np.deg2rad(revolute_thresh) and fine_type_loss == 0:
                num_revolute_success += 1
                revolute_pos_err_coarse.append(coarse_pos_err)
                revolute_angle_err_coarse.append(coarse_angle_err)
                revolute_pos_err_fine.append(fine_pos_err)
                revolute_angle_err_fine.append(fine_angle_err)
    
    assert num_prismatic + num_revolute == num_exp
    print(prismatic_angle_err_coarse)
    # 打印结果
    print(f"Printing results under pris_thresh={prismatic_thresh} m and rev_thresh={revolute_thresh} degree")
    print(f"Reconstruct Success Rate: {num_prismatic_success + num_revolute_success} / {num_exp} = {(num_prismatic_success + num_revolute_success) / num_exp:.4f}")
    
    # 具体打印下两种关节的成功率
    print("Detailed Results: ")
    print(f"prismatic success rate: {num_prismatic_success}/{num_prismatic}, {num_prismatic_success/max(num_prismatic, 1):.4f}")
    prismatic_angle_err_coarse_mean = sum(prismatic_angle_err_coarse) / max(num_prismatic_success, 1)
    prismatic_angle_err_fine_mean = sum(prismatic_angle_err_fine) / max(num_prismatic_success, 1)
    print(f"\tcoarse angle loss: {np.rad2deg(prismatic_angle_err_coarse_mean):.4f} degree")
    print(f"\tfine   angle loss: {np.rad2deg(prismatic_angle_err_fine_mean):.4f} degree") 

    print(f"revolute success rate: {num_revolute_success}/{num_revolute}, {num_revolute_success/max(num_revolute, 1):.4f}")
    revolute_angle_err_coarse_mean = sum(revolute_angle_err_coarse) / max(num_revolute_success, 1)
    revolute_angle_err_fine_mean = sum(revolute_angle_err_fine) / max(num_revolute_success, 1)
    print(f"\tcoarse angle loss: {np.rad2deg(revolute_angle_err_coarse_mean):.4f} degree")
    print(f"\tfine   angle loss: {np.rad2deg(revolute_angle_err_fine_mean):.4f} degree")

    revolute_pos_err_coarse_mean = sum(revolute_pos_err_coarse) / max(num_revolute_success, 1)
    revolute_pos_err_fine_mean = sum(revolute_pos_err_fine) / max(num_revolute_success, 1)
    print(f"\tcoarse pose loss: {revolute_pos_err_coarse_mean * 100:.4f} cm")
    print(f"\tfine   pose loss: {revolute_pos_err_fine_mean * 100:.4f} cm")
    
    # 再分析下失败的那些, 到底是因为什么原因失败的
    # TODO
    
    
# def summary_recon(saved_result, task_yaml):
    # pris_thrs = [0.001, 0.005, 0.01, 0.01, 0.05, 0.1]
    # revo_thrs = [0.5, 1, 5, 10, 10, 20]
    # # 0.05 10 似乎是一个不错的
    
    # # 将数据改为可读取的形式
    # results = [saved_result[k]["recon"] for k in saved_result.keys()]
    # results = [saved_result[k]["recon"] for k in saved_result.keys() if "recon" in saved_result[k].keys()]
    # for i in range(len(pris_thrs)):
    #     print_recon_result(results, pris_thrs[i], revo_thrs[i])
    # print_recon_result(results, 0.05, 10)

############### summary_manipulate ###############



if __name__ == "__main__":
    # summary_explore(saved_result, task_yaml)
    # summary_recon(saved_result, task_yaml)
    print_recon_result(saved_result, 0.05, 10)