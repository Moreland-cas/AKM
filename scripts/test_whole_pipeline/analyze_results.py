import os
import copy
import sys
import json
import yaml
import heapq 
from collections import defaultdict
import numpy as np
from embodied_analogy.project_config import (
    PRISMATIC_JOINT_MAX_RANGE,
    REVOLUTE_JOINT_MAX_RANGE
)

############### utils ###############
class MinKNumbers:
    def __init__(self):
        """
        初始化一个空的数字存储容器
        """
        self.numbers = []
    
    def add_number(self, num):
        """
        添加一个数字到存储容器中
        :param num: 要添加的数字
        """
        self.numbers.append(num)
    
    def add_numbers(self, nums):
        """
        批量添加多个数字到存储容器中
        :param nums: 包含多个数字的可迭代对象
        """
        self.numbers.extend(nums)
    
    def get_min_k(self, k):
        """
        返回存储的数字中最小的k个数字
        :param k: 要返回的最小数字的数量
        :return: 包含最小k个数字的列表
        """
        if k <= 0:
            return []
        if k >= len(self.numbers):
            return sorted(self.numbers.copy())
        
        # 使用堆结构来高效获取最小的k个数字
        return heapq.nsmallest(k, self.numbers)
    
    def clear(self):
        """
        清空存储的所有数字
        """
        self.numbers.clear()
    
    def __str__(self):
        return f"MinKNumbers(currently storing {len(self.numbers)} numbers)"
    
    
def print_array(array, prefix=""):
    # 将一个 array 按照百分比的方式打印出来
    total = sum(array)
    percentages = [f"{(x / total) * 100:.0f}%" for x in array]
    array_str = [str(x) for x in array]
    print(f"{prefix}{'/'.join(percentages)} ({'/'.join(array_str)})")
    
#############################################


############### summary_explore ###############
# 直接把一些判断是否成功的超参放到这里, 方便修改
EXPLORE_PRISMATIC_VALID = 0.05 # m
EXPLORE_REVOLUTE_VALID = 5 # degree

def explore_actually_valid(explore_result):
    """
    输入的是一个 has_valid_explore 为 True 的 explore_result, 本函数判断该 result 是否是真的 valid
    """
    joint_type = explore_result["joint_type"]
    joint_delta = explore_result["joint_state_end"] - explore_result["joint_state_start"]
    
    if joint_type == "prismatic":
        return joint_delta >= EXPLORE_PRISMATIC_VALID
    
    elif joint_type == "revolute":
        return np.rad2deg(joint_delta) >= EXPLORE_REVOLUTE_VALID


def summary_explore(saved_result, task_yaml, verbose=False, num_bins=4):
    """分析探索阶段的结果，包括按初始关节状态分区的成功率"""
    max_tries = task_yaml["explore_env_cfg"]["max_tries"]
    exp_folder = task_yaml["exp_cfg"]["exp_folder"]
    
    # 定义关节范围
    prismatic_joint_range = [0, PRISMATIC_JOINT_MAX_RANGE]
    revolute_joint_range = [0, np.deg2rad(REVOLUTE_JOINT_MAX_RANGE)]
    
    # 创建区间划分
    prismatic_bins = np.linspace(prismatic_joint_range[0], prismatic_joint_range[1], num_bins+1)
    revolute_bins = np.linspace(revolute_joint_range[0], revolute_joint_range[1], num_bins+1)
    
    # 初始化统计数据结构
    num_total_exp = len(saved_result.keys())
    num_total_tries_for_success_exp = 0
    num_total_tries_for_all_exp = 0
    
    # 全局成功率统计
    num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
    num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}
    
    # 按关节类型统计
    num_pred_valid_prismatic = {i: 0 for i in range(1, max_tries + 1)}
    num_actual_valid_prismatic = {i: 0 for i in range(1, max_tries + 1)}
    num_pred_valid_revolute = {i: 0 for i in range(1, max_tries + 1)}
    num_actual_valid_revolute = {i: 0 for i in range(1, max_tries + 1)}
    
    # 按初始关节状态分区统计
    bin_success_rates = defaultdict(list)
    bin_counts = defaultdict(int)  # 记录每个区间的实验数量
    
    # 最小变化统计
    prismatic_heap = MinKNumbers()
    revolute_heap = MinKNumbers()
    
    # 关节类型计数
    total_prismatic = 0
    total_revolute = 0

    # 处理每个任务的结果
    results = [(k, saved_result[k]["explore"]) for k in saved_result.keys()]
    for key, result in results:
        has_valid_explore = result["has_valid_explore"]
        joint_type = result["joint_type"]
        
        # 更新关节类型计数
        if joint_type == "prismatic":
            total_prismatic += 1
        elif joint_type == "revolute":
            total_revolute += 1
            
        # 读取该任务的具体YAML文件获取初始关节状态
        yaml_path = os.path.join(exp_folder, str(key), f"{key}.yaml")
        # if not os.path.exists(yaml_path):
        #     print(f"Warning: YAML file not found for key {key}: {yaml_path}")
        #     continue
            
        # try:
        with open(yaml_path, "r") as f:
            specific_yaml = yaml.safe_load(f)
        load_joint_state = specific_yaml["obj_env_cfg"]["load_joint_state"]
        # except Exception as e:
        #     print(f"Error loading YAML for key {key}: {str(e)}")
        #     continue
        
        # 确定初始状态所属的区间
        bin_key = None
        if joint_type == "prismatic":
            bin_idx = np.digitize(load_joint_state, prismatic_bins) - 1
            bin_key = f"prismatic_bin_{bin_idx}"
        elif joint_type == "revolute":
            bin_idx = np.digitize(load_joint_state, revolute_bins) - 1
            bin_key = f"revolute_bin_{bin_idx}"
        
        # 记录该实验是否成功
        actual_valid = False
        if has_valid_explore:
            actual_valid = explore_actually_valid(result)
        
        # 更新分区统计
        # if bin_key is not None:
        bin_success_rates[bin_key].append(1 if actual_valid else 0)
        bin_counts[bin_key] += 1
        
        # 统计一下所有实验的尝试次数
        if "num_tries" not in result:
            # 对于在 explore 阶段遇到 exception 的 exp, 可能 num_tries 还未写入, 此时默认计算为 25
            result["num_tries"] = task_yaml["explore_env_cfg"]["max_tries"]
        num_tries = result["num_tries"]
        num_total_tries_for_all_exp += num_tries
        
        # 只统计有有效探索的实验
        if not has_valid_explore:
            continue
        
        # 更新成功的那些实验的平均尝试次数
        num_total_tries_for_success_exp += num_tries
        
        # 更新尝试次数统计
        for i in range(num_tries, max_tries + 1):
            num_pred_valid[i] += 1
            if actual_valid:
                num_actual_valid[i] += 1
            
            # 更新按关节类型的统计
            if joint_type == "prismatic":
                num_pred_valid_prismatic[i] += 1
                if actual_valid:
                    num_actual_valid_prismatic[i] += 1
            elif joint_type == "revolute":
                num_pred_valid_revolute[i] += 1
                if actual_valid:
                    num_actual_valid_revolute[i] += 1
        
        # 更新最小变化统计
        if verbose:
            joint_delta = result["joint_state_end"] - result["joint_state_start"]
            if joint_type == "prismatic":
                prismatic_heap.add_number(abs(joint_delta))
            elif joint_type == "revolute":
                revolute_heap.add_number(np.rad2deg(abs(joint_delta)))
        
    # 打印总体结果
    print("\n************** Explore Stage Analysis **************")
    print(f"Success Rate (actual): {num_actual_valid[max_tries]} / {num_total_exp} = {(num_actual_valid[max_tries] / num_total_exp * 100):.2f}%")
    print(f"Success Rate (pred): {num_pred_valid[max_tries]} / {num_total_exp} = {(num_pred_valid[max_tries] / num_total_exp * 100):.2f}%")
    
    # print(f"Average tries (pred): {num_total_tries} / {num_pred_valid[max_tries]} = {num_total_tries / num_pred_valid[max_tries]:.2f}")
    print(f"Average tries (all): {num_total_tries_for_all_exp} / {num_total_exp} = {num_total_tries_for_all_exp / num_total_exp:.2f}")
    
    # 分别打印prismatic和revolute的结果
    if total_prismatic > 0:
        prismatic_actual_rate = num_actual_valid_prismatic[max_tries] / total_prismatic * 100
        prismatic_pred_rate = num_pred_valid_prismatic[max_tries] / total_prismatic * 100
        print(f"\nPrismatic Joints (n={total_prismatic}):")
        print(f"  Actual Success: {prismatic_actual_rate:.2f}%")
        print(f"  Predicted Success: {prismatic_pred_rate:.2f}%")
    
    if total_revolute > 0:
        revolute_actual_rate = num_actual_valid_revolute[max_tries] / total_revolute * 100
        revolute_pred_rate = num_pred_valid_revolute[max_tries] / total_revolute * 100
        print(f"\nRevolute Joints (n={total_revolute}):")
        print(f"  Actual Success: {revolute_actual_rate:.2f}%")
        print(f"  Predicted Success: {revolute_pred_rate:.2f}%")
    
    # 打印详细尝试次数结果
    print("\nDetailed Results (All Joints):")
    for tries in range(1, max_tries + 1):
        success_rate_pred = num_pred_valid[tries] / num_total_exp
        success_rate_actual = num_actual_valid[tries] / num_total_exp
        print(f"tries={tries} | success rate (actual): {(success_rate_actual * 100):.2f}% | success rate (pred): {(success_rate_pred * 100):.2f}%")
    
    # 打印分区统计信息
    print("\nBin Success Rates:")
    for bin_key in sorted(bin_success_rates.keys()):
        success_list = bin_success_rates[bin_key]
        if success_list:
            success_rate = sum(success_list) / len(success_list) * 100
            print(f"{bin_key}: {success_rate:.2f}% ({len(success_list)} samples)")
    
    # 绘制初始关节状态分区的箱形图
    # if bin_success_rates:
    #     plt.figure(figsize=(14, 7))
        
    #     # 准备箱形图数据
    #     bin_labels = []
    #     success_data = []
        
    #     # 处理prismatic分区
    #     for i in range(num_bins):
    #         bin_key = f"prismatic_bin_{i}"
    #         if bin_key in bin_success_rates and bin_success_rates[bin_key]:
    #             bin_labels.append(f"P-Bin{i+1}")
    #             success_data.append(bin_success_rates[bin_key])
        
    #     # 处理revolute分区
    #     for i in range(num_bins):
    #         bin_key = f"revolute_bin_{i}"
    #         if bin_key in bin_success_rates and bin_success_rates[bin_key]:
    #             bin_labels.append(f"R-Bin{i+1}")
    #             success_data.append(bin_success_rates[bin_key])
        
    #     # 绘制箱形图
    #     plt.boxplot(success_data, labels=bin_labels)
    #     plt.title("Success Rate Distribution by Initial Joint State Bins")
    #     plt.ylabel("Success Rate (1=Success, 0=Failure)")
    #     plt.xlabel("Joint State Bins (P=Prismatic, R=Revolute)")
    #     plt.xticks(rotation=45)
    #     plt.grid(True, linestyle='--', alpha=0.7)
        
    #     # 添加分区边界信息
    #     prismatic_bin_edges = [f"{x:.2f}" for x in prismatic_bins]
    #     revolute_bin_edges_deg = [f"{np.rad2deg(x):.1f}" for x in revolute_bins]
        
    #     plt.figtext(0.1, 0.02, f"Prismatic bins (m): {', '.join(prismatic_bin_edges)}", fontsize=9)
    #     plt.figtext(0.1, 0.00, f"Revolute bins (deg): {', '.join(revolute_bin_edges_deg)}", fontsize=9)
        
    #     plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部文本留出空间
    #     plt.show()
    
    # 打印最小变化统计
    if verbose:
        print("\nJoint delta statistic (for pred as valid):")
        print("Prismatic:", prismatic_heap.get_min_k(10))
        print("Revolute:", revolute_heap.get_min_k(10))
    
    print("****************************************************\n")

############### summary_reconstruct ###############
# PIVOT_THRESH = 0.05 # m
# ANGLE_THRESH = 10 # degree

PIVOT_THRESH = 0.03 # m
ANGLE_THRESH = 5 # degree

def is_reconstruct_valid(explore_result, recon_result):
    """
    判断 reconstruct 是否有效, 
    返回:
        joint_type: prismatic/revolute 
        recon_valid: True/False
        
        如果 recon_valid 是 True 的话, 返回 recon_loss
            pivot_loss
            angle_loss
            
        如果 recon_valid 是 False 的话, 返回失败的原因, 一个 np.array([3])
            think_explore_invalid
            think_explore_valid_actually_not
            type_loss == 1
            type_loss == 0, but recon_loss is high
    """
    joint_type = explore_result["joint_type"]
    recon_valid_c = False
    recon_valid_f = False
    failed_reason = np.array([0, 0, 0, 0])
    pivot_loss_c, angle_loss_c = None, None
    pivot_loss_f, angle_loss_f = None, None
    
    # 如果都没有 explore_think_valid, 直接返回
    has_valid_recon = recon_result["has_valid_recon"]
    
    if not has_valid_recon:
        recon_valid_c = False
        recon_valid_f = False
        failed_reason[0] = 1
        return recon_valid_c, recon_valid_f, failed_reason, joint_type, None
    
    # NOTE 接下来的应该都是有 recon_result 的
    # 读取重建结果
    joint_type = recon_result["gt_w"]["joint_type"]
    coarse_loss = recon_result['coarse_loss']
    fine_loss = recon_result['fine_loss']
    
    coarse_type_loss = coarse_loss['type_err']
    angle_loss_c = coarse_loss['angle_err']
    pivot_loss_c = coarse_loss['pos_err']
    
    fine_type_loss = fine_loss['type_err']
    angle_loss_f = fine_loss['angle_err']
    pivot_loss_f = fine_loss['pos_err']
    
    # 根据重建结果判断 recon_valid
    if joint_type == "prismatic":
        recon_valid_c = (angle_loss_c < np.deg2rad(ANGLE_THRESH) and coarse_type_loss == 0)
        recon_valid_f = (angle_loss_f < np.deg2rad(ANGLE_THRESH) and fine_type_loss == 0)
    else:
        recon_valid_c = (pivot_loss_c < PIVOT_THRESH and angle_loss_c < np.deg2rad(ANGLE_THRESH) and coarse_type_loss == 0)
        recon_valid_f = (pivot_loss_f < PIVOT_THRESH and angle_loss_f < np.deg2rad(ANGLE_THRESH) and fine_type_loss == 0)
    
    if recon_valid_f:
        return recon_valid_c, recon_valid_f, failed_reason, joint_type, (pivot_loss_c, pivot_loss_f, angle_loss_c, angle_loss_f)
    
    # 如果到了这里还没有退出, 说明 recon_valid 是 False, 接下来判断是因为什么原因失败的
    assert recon_valid_f == False
    
    # 认为有 valid_explore, 但是实际上没有
    if not explore_actually_valid(explore_result):
        failed_reason[1] = 1
        return recon_valid_c, recon_valid_f, failed_reason, joint_type, None
    
    # NOTE 实际上有 valid_explore, 只是重建的问题
    
    # 关节类型重建错误
    if recon_result["coarse_loss"]["type_err"] > 0:
        failed_reason[2] = 1
        return recon_valid_c, recon_valid_f, failed_reason, joint_type, None
    
    # 关节类型重建对了, 但是优化有问题
    if recon_result["coarse_loss"]["type_err"] == 0:                
        failed_reason[3] = 1
        return recon_valid_c, recon_valid_f, failed_reason, joint_type, None


def summary_recon(saved_result):
    """
    分析一下重建失败的原因:
    1) 来自 explore 的问题
        1.1) 要么是算法直接认为没有有效 explore, 跳过了 reconstruct
        1.2) 要么是算法认为有有效 explore, 但其实没有 (这部分的数量需要单独统计一下)
        
    2) 来自 reconstruct 的问题, 即确实有有效的 explore, 但重建的 joint_type 或者 joint_param 还是出错了
        2.1) 其中有多少 type_loss
    
    ADDITIONAL: 可以绘制一下所有算法认为有有效 explore 的样本中, type_loss 对于 joint_delta 的变化, 以及 recon_loss 对于 joint_delta 的变化
    
    (NOTE 因为那些 type_loss 不为 0 的定量 loss 是没有意义的)
    """
    
    explore_results = []
    recon_results = []
    for k, v in saved_result.items():
        explore_results.append(v["explore"])
        recon_results.append(v["recon"])
            
    num_exp = len(saved_result.keys())
    num_prismatic = 0
    num_revolute = 0
    
    # 先根据 "什么样的 reconstruct 是好的", 计算整体的成功率, 以及成功的那些的误差
    num_prismatic_success_f = 0
    num_revolute_success_f = 0
    
    num_prismatic_success_c = 0
    num_revolute_success_c = 0
    
    prismatic_angle_err_coarse = []
    revolute_angle_err_coarse = []
    revolute_pos_err_coarse = []
    
    prismatic_angle_err_fine = []
    revolute_angle_err_fine = []
    revolute_pos_err_fine = []
    
    # 统计所有的任务, 根据 recon_success_thresh 计算 success_rate
    prismatic_failed_reason_array = np.array([0, 0, 0, 0])
    revolute_failed_reason_array = np.array([0, 0, 0, 0])
    
    for explore_result, recon_result in zip(explore_results, recon_results):
        # 这里的 recon_valid 指 recon 的误差小于一定值
        # failed_reason 是针对 recon_valid_f 判断的
        recon_valid_c, recon_valid_f, failed_reason, joint_type, loss = is_reconstruct_valid(explore_result, recon_result)
        
        num_prismatic += int(joint_type == "prismatic")
        num_revolute += int(joint_type == "revolute")
        
        if recon_valid_c:
            if joint_type == "prismatic":
                num_prismatic_success_c += 1
            else:
                num_revolute_success_c += 1
            
        if recon_valid_f:
            pivot_loss_c, pivot_loss_f, angle_loss_c, angle_loss_f = loss
            if joint_type == "prismatic":
                num_prismatic_success_f += 1
                prismatic_angle_err_coarse.append(angle_loss_c)
                prismatic_angle_err_fine.append(angle_loss_f)
            else:
                num_revolute_success_f += 1
                revolute_pos_err_coarse.append(pivot_loss_c)
                revolute_angle_err_coarse.append(angle_loss_c)
                revolute_pos_err_fine.append(pivot_loss_f)
                revolute_angle_err_fine.append(angle_loss_f)
        else:
            if joint_type == "prismatic":
                prismatic_failed_reason_array += failed_reason
            else:
                revolute_failed_reason_array += failed_reason
    
    assert num_prismatic + num_revolute == num_exp
    # 打印结果
    print("\n************** Reconstruct Stage Analysis **************")
    
    print(f"Success Rate (fine): {num_prismatic_success_f + num_revolute_success_f} / {num_exp} = {((num_prismatic_success_f + num_revolute_success_f) / num_exp * 100):.2f}%")
    print(f"Success Rate (coarse): {num_prismatic_success_c + num_revolute_success_c} / {num_exp} = {((num_prismatic_success_c + num_revolute_success_c) / num_exp * 100):.2f}%")
    
    print(f"Success Rate (fine-prismatic): {num_prismatic_success_f} / {num_prismatic} = {(num_prismatic_success_f / num_prismatic * 100):.2f}%")
    print(f"Success Rate (fine-revolute): {num_revolute_success_f} / {num_revolute} = {(num_revolute_success_f / num_revolute * 100):.2f}%")
    
    print(f"Success Rate (coarse-prismatic): {num_prismatic_success_c} / {num_prismatic} = {(num_prismatic_success_c / num_prismatic * 100):.2f}%")
    print(f"Success Rate (coarse-revolute): {num_revolute_success_c} / {num_revolute} = {(num_revolute_success_c / num_revolute * 100):.2f}%")
    print(f"note: Reconstruction loss under PIVOT_THRESH: {PIVOT_THRESH} m and ANGLE_THRESH: {ANGLE_THRESH} degree is considered valid.")
    
    # 具体打印下两种关节的成功率
    print("\nDetailed Results: ")
    print(f"PRISMATIC: \n\t{num_prismatic} in total")
    print(f"Success/invalid_explore/false_positive/wrong_type/bad_optimize: ")
    print_array(prefix="\t", array=[num_prismatic_success_f, *prismatic_failed_reason_array])
    
    prismatic_angle_err_coarse_mean_rad = sum(prismatic_angle_err_coarse) / max(num_prismatic_success_f, 1)
    prismatic_angle_err_coarse_mean_degree = np.rad2deg(prismatic_angle_err_coarse_mean_rad)
    
    prismatic_angle_err_fine_mean_rad = sum(prismatic_angle_err_fine) / max(num_prismatic_success_f, 1)
    prismatic_angle_err_fine_mean_degree = np.rad2deg(prismatic_angle_err_fine_mean_rad)
    
    print(f"Angle loss: \n\t{prismatic_angle_err_coarse_mean_degree:.2f} degree -> {prismatic_angle_err_fine_mean_degree:.2f} degree (coarse -> fine)")

    print(f"\nREVOLUTE: \n\t{num_revolute} in total")
    print(f"Success/invalid_explore/false_positive/wrong_type/bad_optimize: ")
    print_array(prefix="\t", array=[num_revolute_success_f, *revolute_failed_reason_array])
    
    revolute_angle_err_coarse_mean_rad = sum(revolute_angle_err_coarse) / max(num_revolute_success_f, 1)
    revolute_angle_err_coarse_mean_degree = np.rad2deg(revolute_angle_err_coarse_mean_rad)
    
    revolute_angle_err_fine_mean_rad = sum(revolute_angle_err_fine) / max(num_revolute_success_f, 1)
    revolute_angle_err_fine_mean_degree = np.rad2deg(revolute_angle_err_fine_mean_rad)
    
    print(f"Angle loss: \n\t{revolute_angle_err_coarse_mean_degree:.2f} degree -> {revolute_angle_err_fine_mean_degree:.2f} degree (coarse -> fine)")

    revolute_pos_err_coarse_mean_m = sum(revolute_pos_err_coarse) / max(num_revolute_success_f, 1)
    revolute_pos_err_coarse_mean_cm = revolute_pos_err_coarse_mean_m * 100
    
    revolute_pos_err_fine_mean_m = sum(revolute_pos_err_fine) / max(num_revolute_success_f, 1)
    revolute_pos_err_fine_mean_cm = revolute_pos_err_fine_mean_m * 100
    
    print(f"Pivot loss: \n\t{revolute_pos_err_coarse_mean_cm:.2f} cm -> {revolute_pos_err_fine_mean_cm:.2f} cm (coarse -> fine)")
    print("****************************************************\n")
    
############### summary_manipulate ###############
# 相对误差在 10 % 内的视为成功, 如打开 10 cm, 最终误差在 1 cm 内的
MANIP_RELATIVE_VALID_THRESH = 0.1


def is_manip_success(manip_result):
    """
    manip_distance: 该任务实际需要 manip 的 delta joint state
    NOTE 对于 revolute 来说, manip_distance 的单位是 degree, 需要转换成 rad
    
    返回的是:
        manip_success: 是否算成功的 manip (误差阈值在 MANIP_RELATIVE_VALID_THRESH * manip_distance 内）
        loss_list: 多次闭环操作后的误差变化, 最少有一个, 最小的 key 为 0, 最大的 key 为 MAX_TRIES
        relative_error: last_loss / manip_distance
    """
    manip_distance = abs(manip_result["manip_end_state"] - manip_result["manip_start_state"])
    
    time_steps = manip_result.keys()
    time_steps = list(time_steps)
    # delete 两个字符串的 key
    try:
        time_steps.remove("manip_start_state")
        time_steps.remove("manip_end_state")
        time_steps.remove("exception")
    except Exception as e:
        pass
    time_steps = [int(time_step) for time_step in time_steps]
    max_time_step = str(max(time_steps))
    last_result = manip_result[max_time_step]
    last_loss = abs(last_result["diff"])
    relative_error = last_loss / manip_distance
    
    loss_list = []
    # TODO 更新后的版本的 manip_result 里不应该有 "-1", 取而代之的是 "0"
    # if "-1" in manip_result.keys():
    #     loss_list.append(abs(manip_result["-1"]["diff"]))
        
    for i in range(int(max_time_step) + 1):
        loss_list.append(abs(manip_result[str(i)]["diff"]))
    
    # if joint_type == "revolute":
    #     manip_distance = np.deg2rad(manip_distance)
        
    if last_loss < MANIP_RELATIVE_VALID_THRESH * manip_distance:
        return True, loss_list, relative_error
    return False, loss_list, relative_error


def print_manip_summary_dict(summary_dict):
    print("prismatic")
    print("\topen")
    for scale in summary_dict["prismatic"]["open"]:
        print(f"\t\tscale {scale}:")
        
        # 确定 sucess 和 failed 的个数
        num_success = summary_dict["prismatic"]["open"][scale]["success"]
        num_fail_recon = summary_dict["prismatic"]["open"][scale]["failed_recon"]
        num_fail_manip = summary_dict["prismatic"]["open"][scale]["failed_manip"]
        print(f"\t\t\tsuccess/fail_recon/fail_manip: {num_success}/{num_fail_recon}/{num_fail_manip}")
        
        for loss_list in summary_dict["prismatic"]["open"][scale]["loss_lists"]:
            loss_list = [f"{loss * 100:.2f}cm" for loss in loss_list]
            print(f"\t\t\t{loss_list}:")
    
    print("\tclose")
    for scale in summary_dict["prismatic"]["close"]:
        print(f"\t\tscale {scale}:")
        
        num_success = summary_dict["prismatic"]["close"][scale]["success"]
        num_fail_recon = summary_dict["prismatic"]["close"][scale]["failed_recon"]
        num_fail_manip = summary_dict["prismatic"]["close"][scale]["failed_manip"]
        print(f"\t\t\tsuccess/fail_recon/fail_manip: {num_success}/{num_fail_recon}/{num_fail_manip}")
        
        for loss_list in summary_dict["prismatic"]["close"][scale]["loss_lists"]:
            loss_list = [f"{loss * 100:.2f}cm" for loss in loss_list]
            print(f"\t\t\t{loss_list}")

    print("revolute")
    print("\topen")
    for scale in summary_dict["revolute"]["open"]:
        print(f"\t\tscale {scale}:")
        
        num_success = summary_dict["revolute"]["open"][scale]["success"]
        num_fail_recon = summary_dict["revolute"]["open"][scale]["failed_recon"]
        num_fail_manip = summary_dict["revolute"]["open"][scale]["failed_manip"]
        print(f"\t\t\tsuccess/fail_recon/fail_manip: {num_success}/{num_fail_recon}/{num_fail_manip}")
        
        for loss_list in summary_dict["revolute"]["open"][scale]["loss_lists"]:
            loss_list = [f"{np.rad2deg(loss):.2f}dg" for loss in loss_list]
            print(f"\t\t\t{loss_list}")

    print("\tclose")
    for scale in summary_dict["revolute"]["close"]:
        print(f"\t\tscale {scale}:")
        
        num_success = summary_dict["revolute"]["close"][scale]["success"]
        num_fail_recon = summary_dict["revolute"]["close"][scale]["failed_recon"]
        num_fail_manip = summary_dict["revolute"]["close"][scale]["failed_manip"]
        print(f"\t\t\tsuccess/fail_recon/fail_manip: {num_success}/{num_fail_recon}/{num_fail_manip}")
        
        for loss_list in summary_dict["revolute"]["close"][scale]["loss_lists"]:
            loss_list = [f"{np.rad2deg(loss):.2f}dg" for loss in loss_list]
            print(f"\t\t\t{loss_list}")


def process_manip_grid_dict(summary_dict, task_yaml, joint_type):
    """
    这里的 grid_dict 包含多个 "i_j" 的 key, 每个 key 下存储一个 base_dict:
    {
        "loss_lists": [],
        "success": 0,
        "failed_manip": 0,
        "failed_recon": 0
    }
    返回一个 matrix 信息, 需要包含：
        Success/Failed-recon/Failed-manip
        #avg_manip
        final_error
        error_traj
    """
    # NOTE 这里 +1 是因为 loss_list 的第一个是未操作的状态, 也进行了存储
    max_tries_plus_one = task_yaml["manip_env_cfg"]["max_manip"] + 1
    
    # NOTE: summary_dict 中存储的 loss 已经是 cm 或者 degree, 但是有正有负
    # unit = "cm" if joint_type == "prismatic" else "degree"
    
    for i_j, base_dict in summary_dict.items():
        num_manip = 0
        # 将所有 loss_list 补全到 max_tries_plus_one 长度, 用原本数组的最后一个元素进行 pad
        for i, loss_list in enumerate(base_dict["loss_lists"]):
            # NOTE 这里需要减去 1
            num_manip += (len(loss_list) - 1)
            # 进行单位转换
            if joint_type == "prismatic":
                loss_list = [abs(loss * 100) for loss in loss_list]
            else:
                loss_list = [abs(np.rad2deg(loss)) for loss in loss_list]
            loss_list = loss_list + [loss_list[-1]] * int(max_tries_plus_one - len(loss_list))
            base_dict["loss_lists"][i] = loss_list
        
        # num_exp * close_loop_times    
        loss_array = np.array(base_dict["loss_lists"]) 
        
        if loss_array.ndim == 1:
            # 处理 num_exp == 1 的情况
            loss_array = loss_array.reshape(1, -1)
        
        # num_success = base_dict["success"]
        # num_failed_recon = base_dict["failed_recon"]
        # num_failed_manip = base_dict["failed_manip"]
        
        # print(f"\tSuccess/Failed-recon/Failed-manip:")
        # print_array(prefix="\t\t", array=[num_success, num_failed_recon, num_failed_manip])
        
        if loss_array.shape[-1] == 0:
            base_dict["avg_manips"] = None
            base_dict["final_error"] = None
            base_dict["error_traj"] = None
            
            base_dict["loss_lists"] = []
            continue
        
        base_dict["avg_manips"] = num_manip / loss_array.shape[0]
        # print(f"\tAvg Manips: {(num_manip / loss_array.shape[0]):.2f}")
        base_dict["final_error"] = loss_array[:, -1].mean()
        # print(f"\tFinal Error: {loss_array[:, -1].mean():.2f} {unit}")
        
        base_dict["error_traj"] = [loss_array[:, i].mean() for i in range(int(max_tries_plus_one))]
        # closed_loop_error_string = "->".join([f"[{i}] {loss_array[:, i].mean():.2f} {unit}" for i in range(int(max_tries_plus_one))])
        # print(f"\tClosed-loop Error: {closed_loop_error_string}")
        
        # base_dict["print_str"] = f'{base_dict["success"]}/{base_dict["failed_recon"]}/{base_dict["failed_manip"]/{base_dict["avg_manips"]}/{base_dict["final_error"]}/{base_dict["error_traj"]}}'
        base_dict["loss_lists"] = []


def print_matrix_string(matrix):
    """
    以网格形式打印二维矩阵中的字符串
    :param matrix: 二维列表，每个元素是字符串
    """
    # 获取矩阵的行数和列数
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    # 打印每行
    for i in range(m):
        # 打印每列
        for j in range(n):
            # 打印元素，使用制表符分隔
            print(f"{matrix[i][j]}", end="\t")
        # 每行结束后换行
        print()
    
                           
def summary_manip(saved_result, task_yaml, verbose=False):
    print("\n************** Manipulation Stage Analysis **************")
    # 针对每个 grid 存储一系列信息
    keys = saved_result[0]["manip"].keys()
    prismatic_grid_dicts = {}
    revolute_grid_dicts = {}
    
    base_dict = {
        # loss_lists 存储 list of loss_traj
        "loss_lists": [],
        "success": 0,
        # 这里将 failed 的分为 重建有问题 和 manip 两类
        "failed_manip": 0,
        "failed_recon": 0
    }
    
    for key in list(keys):
        # 对于每个 grid, 维护一个 success_dict
        prismatic_grid_dicts[key] = copy.deepcopy(base_dict)
        revolute_grid_dicts[key] = copy.deepcopy(base_dict)
    
    # 统计一下整体的成功率
    num_total_exp = len(saved_result.keys()) * len(saved_result[0]["manip"].keys())
    num_success_manip = 0
    relative_error_list = []
    
    for task_id, v in saved_result.items():
        
        explore_result = v["explore"]
        recon_result = v["recon"]
        joint_type = explore_result["joint_type"]
        
        grid_dicts = prismatic_grid_dicts if joint_type == "prismatic" else revolute_grid_dicts
        
        # 确定任务类型
        # yaml_path = os.path.join(log_folder, str(task_id), f"{task_id}.yaml")
        # with open(yaml_path, "r") as f:
        #     yaml_dict = yaml.safe_load(f)

        for key, manip_result in v["manip"].items():
            # 判断 manip 是否成功
            manip_success, loss_list, relative_error = is_manip_success(manip_result)
            if manip_success:
                num_success_manip += 1
                # 在这里再统计一下平均相对误差
                relative_error_list.append(relative_error)
                
                grid_dicts[key]["success"] += 1
                grid_dicts[key]["loss_lists"].append(loss_list)

            else:
                # 判断是哪种原因导致的 manip 失败
                if is_reconstruct_valid(explore_result, recon_result)[0]:
                    grid_dicts[key]["failed_manip"] += 1
                else:
                    grid_dicts[key]["failed_recon"] += 1
            
    print(f"Success Rate: {num_success_manip}/{num_total_exp} = {(num_success_manip / num_total_exp * 100):.2f}%")
    print(f"Relative Error (mean): {(sum(relative_error_list) / len(relative_error_list) * 100):.2f}%")
    
    # 对于 prismatic_grid_dicts 和 revolute_grid_dicts 进行处理
    process_manip_grid_dict(prismatic_grid_dicts, task_yaml, joint_type="prismatic")
    process_manip_grid_dict(revolute_grid_dicts, task_yaml, joint_type="revolute")
    
    # TODO print dict
    print("Revolute manip summary:")
    for k, v in revolute_grid_dicts.items():
        print(k)
        print(v)
    # print(revolute_grid_dicts)
    
    print("Prismatic manip summary:")
    for k, v in prismatic_grid_dicts.items():
        print(k)
        print(v)
    # print(prismatic_grid_dicts)
    print("****************************************************\n")
    

def analyze_and_save(run_name):
    task_yaml_path = f"/home/zby/Programs/Embodied_Analogy/cfgs/base_{run_name}.yaml"
    
    with open(task_yaml_path, "r") as f:
        task_yaml = yaml.safe_load(f)
    # "/home/zby/Programs/Embodied_Analogy/assets/logs/test"
    log_folder = task_yaml["exp_cfg"]["exp_folder"]

    saved_result = {}
    run_folders = os.listdir(log_folder)
    for run_folder in run_folders:
        run_path = os.path.join(log_folder, run_folder)
        
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
        
    # 这里要确保每个 run 的结果都读取进来了
    assert len(saved_result.keys()) == len(run_folders)

    save_analysis_path = f"/home/zby/Programs/Embodied_Analogy/analysis/{run_name}.txt"
    with open(save_analysis_path, "w") as f:
        sys.stdout = f
        summary_explore(saved_result, task_yaml)
        summary_recon(saved_result)
        summary_manip(saved_result, task_yaml)
        
    
if __name__ == "__main__":
    # run_name = "6_4"
    # run_name = "6_6"
    # run_name = "6_8"
    # run_name = "6_10"
    # run_name = "6_11"
    # run_name = "6_12"
    # run_name = "6_17"
    # run_name = "6_18"
    # run_name = "6_20"
    # run_name = "6_21"
    # run_name = "6_22"
    # run_name = "6_23"
    # run_name = "6_24"
    # run_name = "6_25"
    # run_name = "6_26"
    # run_name = "6_27"
    names = [
        # "6_21",
        # "6_18",
        # "6_26",
        # "6_27",
        # "6_17"
        "6_30"
    ]
    for name in names:
        analyze_and_save(name)
    
        