import os
import sys
import json
import yaml
import heapq 
import numpy as np

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

# 写的时候直接把函数的输入改为 saved_result 这个 dict, 方便假如我们只想跑 explore 以及 explore 的 summary
def summary_explore(saved_result, task_yaml, verbose=False):
    max_tries = task_yaml["explore_env_cfg"]["max_tries"]

    # 到此为止把所有有 result 的读取进来了
    num_total_exp = len(saved_result.keys()) # 所有的实验
    
    # NOTE: num_total_tries 记录了对于那些 pred_valid 的案例一共使用了多少次尝试
    num_total_tries = 0

    # 记录下 tries 从 1 到 MAX_TRIES 个的 pred_valid 个数和 actual_valid 个数
    num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
    num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}

    # 用于记录实际的状态变化最小的 k 个, 用于确定 explore valid 的阈值
    prismatic_heap = MinKNumbers()
    revolute_heap = MinKNumbers()

    results = [saved_result[k]["explore"] for k in saved_result.keys()]
    # results = [saved_result[k]["explore"] for k in saved_result.keys() if "explore" in saved_result[k].keys()]
    for i, result in enumerate(results):
        has_valid_explore = result["has_valid_explore"]
        
        if not has_valid_explore:
            continue
        
        num_tries = result["num_tries"]
        num_total_tries += num_tries
        
        if verbose:
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
    print("\n************** Explore Stage Analysis **************")
    
    print(f"Success Rate (actual): {num_actual_valid[max_tries]} / {num_total_exp} = {(num_actual_valid[max_tries] / num_total_exp * 100):.2f}%")
    print(f"Success Rate (pred): {num_pred_valid[max_tries]} / {num_total_exp} = {(num_pred_valid[max_tries] / num_total_exp * 100):.2f}%")
    print(f"Average tries (pred): {num_total_tries} / {num_pred_valid[max_tries]} = {num_total_tries / num_pred_valid[max_tries]:.2f}")

    if verbose:
        print("jonit delta statistic (for pred as valid):")
        print("prismatic:", prismatic_heap.get_min_k(10))
        print("revolute:", revolute_heap.get_min_k(10))

    # 打印每个尝试次数的成功率
    print("Detailed Results:")
    for tries in range(1, max_tries + 1):
        success_rate_pred = num_pred_valid[tries] / num_total_exp
        success_rate_actual = num_actual_valid[tries] / num_total_exp
        print(f"tries={tries} | success rate (actual): {(success_rate_actual * 100):.2f}% | success rate (pred): {(success_rate_pred * 100):.2f}%")
    
    print("****************************************************\n")


############### summary_reconstruct ###############
PIVOT_THRESH = 0.05 # m
ANGLE_THRESH = 10 # degree

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
    recon_valid = False
    failed_reason = np.array([0, 0, 0, 0])
    pivot_loss_c, angle_loss_c = None, None
    pivot_loss_f, angle_loss_f = None, None
    
    # 如果都没有 explore_think_valid, 直接返回
    # TODO 记得把这里改回去
    if "has_valid_recon" in recon_result:
        has_valid_recon = recon_result["has_valid_recon"]
    elif "has_valid_reconstruct" in recon_result:
        has_valid_recon = recon_result["has_valid_reconstruct"]
    else:
        assert "no has_valid_recon or has_valid_reconstruct in recon_result"
    
    if not has_valid_recon:
        recon_valid = False
        failed_reason[0] = 1
        return recon_valid, failed_reason, joint_type, None
    
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
        if angle_loss_f < np.deg2rad(ANGLE_THRESH) and fine_type_loss == 0:
            recon_valid = True
            return recon_valid, failed_reason, joint_type, (pivot_loss_c, pivot_loss_f, angle_loss_c, angle_loss_f)
    
    if joint_type == "revolute":
        if pivot_loss_f < PIVOT_THRESH and angle_loss_f < np.deg2rad(ANGLE_THRESH) and fine_type_loss == 0:
            recon_valid = True
            return recon_valid, failed_reason, joint_type, (pivot_loss_c, pivot_loss_f, angle_loss_c, angle_loss_f)
    
    # 如果到了这里还没有退出, 说明 recon_valid 是 False, 接下来判断是因为什么原因失败的
    assert recon_valid == False
    
    # 认为有 valid_explore, 但是实际上没有
    if not explore_actually_valid(explore_result):
        recon_valid = False
        failed_reason[1] = 1
        return recon_valid, failed_reason, joint_type, None
    
    # NOTE 实际上有 valid_explore, 只是重建的问题
    
    # 关节类型重建错误
    if recon_result["coarse_loss"]["type_err"] > 0:
        recon_valid = False
        failed_reason[2] = 1
        return recon_valid, failed_reason, joint_type, None
    
    # 关节类型重建对了, 但是优化有问题
    if recon_result["coarse_loss"]["type_err"] == 0:                
        recon_valid = False
        failed_reason[3] = 1
        return recon_valid, failed_reason, joint_type, None


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
    num_prismatic_success = 0
    num_revolute_success = 0
    
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
        recon_valid, failed_reason, joint_type, loss = is_reconstruct_valid(explore_result, recon_result)
        
        num_prismatic += int(joint_type == "prismatic")
        num_revolute += int(joint_type == "revolute")
        
        if recon_valid:
            pivot_loss_c, pivot_loss_f, angle_loss_c, angle_loss_f = loss
            if joint_type == "prismatic":
                num_prismatic_success += 1
                prismatic_angle_err_coarse.append(angle_loss_c)
                prismatic_angle_err_fine.append(angle_loss_f)
            else:
                num_revolute_success += 1
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
    
    print(f"Success Rate: {num_prismatic_success + num_revolute_success} / {num_exp} = {((num_prismatic_success + num_revolute_success) / num_exp * 100):.2f}%")
    print(f"note: Reconstruction loss under PIVOT_THRESH: {PIVOT_THRESH} m and ANGLE_THRESH: {ANGLE_THRESH} degree is considered valid.")
    
    # 具体打印下两种关节的成功率
    print("\nDetailed Results: ")
    print(f"PRISMATIC: \n\t{num_prismatic} in total")
    print(f"Success/invalid_explore/false_positive/wrong_type/bad_optimize: ")
    print_array(prefix="\t", array=[num_prismatic_success, *prismatic_failed_reason_array])
    
    prismatic_angle_err_coarse_mean_rad = sum(prismatic_angle_err_coarse) / max(num_prismatic_success, 1)
    prismatic_angle_err_coarse_mean_degree = np.rad2deg(prismatic_angle_err_coarse_mean_rad)
    
    prismatic_angle_err_fine_mean_rad = sum(prismatic_angle_err_fine) / max(num_prismatic_success, 1)
    prismatic_angle_err_fine_mean_degree = np.rad2deg(prismatic_angle_err_fine_mean_rad)
    
    print(f"Angle loss: \n\t{prismatic_angle_err_coarse_mean_degree:.2f} degree -> {prismatic_angle_err_fine_mean_degree:.2f} degree (coarse -> fine)")

    print(f"\nREVOLUTE: \n\t{num_revolute} in total")
    print(f"Success/invalid_explore/false_positive/wrong_type/bad_optimize: ")
    print_array(prefix="\t", array=[num_revolute_success, *revolute_failed_reason_array])
    
    revolute_angle_err_coarse_mean_rad = sum(revolute_angle_err_coarse) / max(num_revolute_success, 1)
    revolute_angle_err_coarse_mean_degree = np.rad2deg(revolute_angle_err_coarse_mean_rad)
    
    revolute_angle_err_fine_mean_rad = sum(revolute_angle_err_fine) / max(num_revolute_success, 1)
    revolute_angle_err_fine_mean_degree = np.rad2deg(revolute_angle_err_fine_mean_rad)
    
    print(f"Angle loss: \n\t{revolute_angle_err_coarse_mean_degree:.2f} degree -> {revolute_angle_err_fine_mean_degree:.2f} degree (coarse -> fine)")

    revolute_pos_err_coarse_mean_m = sum(revolute_pos_err_coarse) / max(num_revolute_success, 1)
    revolute_pos_err_coarse_mean_cm = revolute_pos_err_coarse_mean_m * 100
    
    revolute_pos_err_fine_mean_m = sum(revolute_pos_err_fine) / max(num_revolute_success, 1)
    revolute_pos_err_fine_mean_cm = revolute_pos_err_fine_mean_m * 100
    
    print(f"Pivot loss: \n\t{revolute_pos_err_coarse_mean_cm:.2f} cm -> {revolute_pos_err_fine_mean_cm:.2f} cm (coarse -> fine)")
    print("****************************************************\n")
    
############### summary_manipulate ###############
# 相对误差在 10 % 内的视为成功, 如打开 10 cm, 最终误差在 1 cm 内的
MANIP_RELATIVE_VALID_THRESH = 0.1

def is_manip_success(joint_type, manip_result, manip_distance):
    """
    manip_distance: 该任务实际需要 manip 的 delta joint state
    NOTE 对于 revolute 来说, manip_distance 的单位是 degree, 需要转换成 rad
    
    返回的是:
        manip_success: 是否算成功的 manip (误差阈值在 MANIP_RELATIVE_VALID_THRESH * manip_distance 内）
        loss_list: 多次闭环操作后的误差变化, 最少有一个, 最小的 key 为 0, 最大的 key 为 MAX_TRIES
        relative_error: last_loss / manip_distance
    """
    time_steps = manip_result.keys()
    time_steps = list(time_steps)
    time_steps = [int(time_step) for time_step in time_steps]
    max_time_step = str(max(time_steps))
    last_result = manip_result[max_time_step]
    last_loss = abs(last_result["diff"])
    relative_error = last_loss / manip_distance
    
    loss_list = []
    # TODO 更新后的版本的 manip_result 里不应该有 "-1", 取而代之的是 "0"
    if "-1" in manip_result.keys():
        loss_list.append(abs(manip_result["-1"]["diff"]))
        
    for i in range(int(max_time_step) + 1):
        loss_list.append(abs(manip_result[str(i)]["diff"]))
    
    if joint_type == "revolute":
        manip_distance = np.deg2rad(manip_distance)
        
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

def process_manip_summary_dict(summary_dict, task_yaml):
    # NOTE 这里 +1 是因为 loss_list 的第一个是未操作的状态, 也进行了存储
    max_tries_plus_one = task_yaml["manip_env_cfg"]["max_manip"] + 1
    
    # NOTE: summary_dict 中存储的 loss 已经是 cm 或者 degree, 但是有正有负
    joint_types = ("prismatic", "revolute")
    manip_types = ("open", "close")
    
    for joint_type in joint_types:
        unit = "cm" if joint_type == "prismatic" else "degree"
        for manip_type in manip_types:
            for scale in summary_dict[joint_type][manip_type]:
                # print(f"{joint_type} {manip_type} {scale}")
                num_manip = 0
                # 将所有 loss_list 补全到 max_tries_plus_one 长度, 用原本数组的最后一个元素进行 pad
                for i, loss_list in enumerate(summary_dict[joint_type][manip_type][scale]["loss_lists"]):
                    # NOTE 这里需要减去 1
                    num_manip += (len(loss_list) - 1)
                    # 进行单位转换
                    if joint_type == "prismatic":
                        loss_list = [abs(loss * 100) for loss in loss_list]
                    else:
                        loss_list = [abs(np.rad2deg(loss)) for loss in loss_list]
                    loss_list = loss_list + [loss_list[-1]] * int(max_tries_plus_one - len(loss_list))
                    summary_dict[joint_type][manip_type][scale]["loss_lists"][i] = loss_list
                
                # num_exp * close_loop_times    
                loss_array = np.array(summary_dict[joint_type][manip_type][scale]["loss_lists"]) 
                
                if loss_array.ndim == 1:
                    # 处理 num_exp == 1 的情况
                    loss_array = loss_array.reshape(1, -1)
                
                if joint_type == "prismatic":
                    scale_str = str(scale * 100)
                else:
                    scale_str = scale
                    
                print(f"\n{manip_type} {scale_str} {unit} ({joint_type}):")
                num_success = summary_dict[joint_type][manip_type][scale]["success"]
                num_failed_recon = summary_dict[joint_type][manip_type][scale]["failed_recon"]
                num_failed_manip = summary_dict[joint_type][manip_type][scale]["failed_manip"]
                
                print(f"\tSuccess/Failed-recon/Failed-manip:")
                print_array(prefix="\t\t", array=[num_success, num_failed_recon, num_failed_manip])
                      
                # print(f"\tSuccess rate: {num_success}/{num_success + num_failed_recon + num_failed_manip}, {num_success / (num_success + num_failed_recon + num_failed_manip) * 100}")
                # print(f"\tfailed-recon: {num_failed_recon}/{num_success + num_failed_recon + num_failed_manip}, {num_failed_recon / (num_success + num_failed_recon + num_failed_manip) * 100}")
                # print(f"\tfailed-manip: {num_failed_manip}/{num_success + num_failed_recon + num_failed_manip}, {num_failed_manip / (num_success + num_failed_recon + num_failed_manip) * 100}")
                print(f"\tAvg Manips: {(num_manip / loss_array.shape[0]):.2f}")
                print(f"\tFinal Error: {loss_array[:, -1].mean():.2f} {unit}")
                
                closed_loop_error_string = "->".join([f"[{i}] {loss_array[:, i].mean():.2f} {unit}" for i in range(int(max_tries_plus_one))])
                print(f"\tClosed-loop Error: {closed_loop_error_string}")
                # for i in range(int(max_tries_plus_one)):
                #     print(f"\tclose loop {i}: {loss_array[:, i].mean():.2f} {unit}")
                       
def summary_manip(saved_result, task_yaml, verbose=False):
    print("\n************** Manipulation Stage Analysis **************")
    # 用一个 dict 来记录
    success_dict = {
        "prismatic": {
            # "open": {"scale": scale_dict}
            # 这里只存储 success 对应的 loss_list
            # scale_dict: {"loss_lists": [], success: 10, failed: 20}
            "open": {}, 
            "close": {},
        },
        "revolute": {
            "open": {}, 
            "close": {},
        },
    }
    # 统计一下整体的成功率
    num_total_exp = len(saved_result.keys())
    num_success_manip = 0
    relative_error_list = []
    
    for task_id, v in saved_result.items():
        
        explore_result = v["explore"]
        recon_result = v["recon"]
        manip_result = v["manip"]
        
        # 确定任务类型
        yaml_path = os.path.join(log_folder, str(task_id), f"{task_id}.yaml")
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
        
        joint_type = explore_result["joint_type"]
        manip_type = yaml_dict["manip_env_cfg"]["manip_type"]
        manip_distance = yaml_dict["manip_env_cfg"]["manip_distance"]
        
        if manip_distance not in success_dict[joint_type][manip_type]:
            success_dict[joint_type][manip_type][manip_distance] = {
                "loss_lists": [],
                "success": 0,
                # 这里将 failed 的分为 重建有问题 和 manip 两类
                "failed_manip": 0,
                "failed_recon": 0
            }
        
        # 判断 manip 是否成功
        manip_success, loss_list, relative_error = is_manip_success(joint_type, manip_result, manip_distance)
        if manip_success:
            num_success_manip += 1
            # 在这里再统计一下平均相对误差
            relative_error_list.append(relative_error)
            
            success_dict[joint_type][manip_type][manip_distance]["success"] += 1
            success_dict[joint_type][manip_type][manip_distance]["loss_lists"].append(loss_list)
        else:
            # 判断是哪种原因导致的 manip 失败
            if is_reconstruct_valid(explore_result, recon_result)[0]:
                success_dict[joint_type][manip_type][manip_distance]["failed_manip"] += 1
            else:
                success_dict[joint_type][manip_type][manip_distance]["failed_recon"] += 1
            
    if verbose:
        print_manip_summary_dict(success_dict)
    
    print(f"Success Rate: {num_success_manip}/{num_total_exp} = {(num_success_manip / num_total_exp * 100):.2f}%")
    print(f"Relative Error (mean): {(sum(relative_error_list) / len(relative_error_list) * 100):.2f}%")
    process_manip_summary_dict(success_dict, task_yaml)
    print("****************************************************\n")
    
    
if __name__ == "__main__":
    run_name = "6_4"
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
        