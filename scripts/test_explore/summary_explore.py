import os
import heapq
import json
import pickle
import numpy as np
import argparse
from embodied_analogy.utility.utils import explore_actually_valid
from embodied_analogy.utility.constants import ASSET_PATH, EXPLORE_PRISMATIC_VALID, EXPLORE_REVOLUTE_VALID


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
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="explore_512", help='Folder where things are stored')
args = parser.parse_args()

# root_path = os.path.join("/media/zby/MyBook1/embody_analogy_data/assets/logs/", args.run_name)
root_path = os.path.join(ASSET_PATH, "logs", args.run_name)
results = []

# 遍历文件夹
max_tries = 0
for object_folder in os.listdir(root_path):
    object_path = os.path.join(root_path, object_folder)
    try:
        # 读取 cfg.json
        # with open(os.path.join(object_path, 'cfg.json'), 'r') as cfg_file:
        #     cfg = cfg_file.read()  # 根据需要解析 JSON
        with open(os.path.join(object_path, 'cfg.json'), 'r', encoding='utf-8') as file:
            cfg = json.load(file)
            max_tries = cfg["max_tries"]

        # 读取 output.txt
        with open(os.path.join(object_path, 'output.txt'), 'r') as output_file:
            output_lines = output_file.readlines()
            output = output_lines[-1]  # 获取最后一行

        # 读取 result.pkl
        with open(os.path.join(object_path, 'result.pkl'), 'rb') as result_file:
            result = pickle.load(result_file)

    except Exception as e:
        print(f"Error while processing {object_path}: {e}")
        continue  # 出现异常时跳过当前文件夹

    # 检查 output 的最后一行
    if not output.startswith("done"):
        print(f"Error: Last line of output in {object_path} does not start with 'done'.")
        continue  # 如果不符合条件，跳过当前文件夹

    # 添加结果到 results 列表
    results.append(result)

# 到此为止把所有有 result 的读取进来了
# 统计有效结果
# max_tries = 10
num_total_exp = len(os.listdir(root_path)) # 所有的实验
num_total_tries = 0

# 记录下 tries 从 1 到 MAX_TRIES 个的 pred_valid 个数和 actual_valid 个数
num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}

# prismatic_joint_delta_min = 1e6
# revolute_joint_delta_min = 1e6
prismatic_heap = MinKNumbers()
revolute_heap = MinKNumbers()
# debug
# prismatic_heap_debug = MinKNumbers()

for i, result in enumerate(results):
    num_tries = result["num_tries"]
    num_total_tries += num_tries
    has_valid_explore = result["has_valid_explore"]
    
    if has_valid_explore:
        joint_delta = result["joint_state_end"] - result["joint_state_start"]
        if result["joint_type"] == "prismatic":
            prismatic_heap.add_number(abs(joint_delta))
        elif result["joint_type"] == "revolute":
            revolute_heap.add_number(abs(joint_delta))
    
    for i in range(num_tries, max_tries + 1):
        if has_valid_explore:
            num_pred_valid[i] += 1
        
            if explore_actually_valid(result):
                num_actual_valid[i] += 1
                
                # debug
                # if result["joint_type"] == "prismatic":
                #     prismatic_heap_debug.add_number(abs(joint_delta))
            
    # 看一下成功的里面的打开角度是多少
    # joint_type = result["joint_type"]
    # joint_delta = result["joint_state_end"] - result["joint_state_start"]
    # if joint_type == "prismatic":
    #     prismatic_joint_delta_min = min(prismatic_joint_delta_min, joint_delta * 100) # cm
        # print(joint_delta * 100, "cm")
    # else:
        # print(np.rad2deg(joint_delta), "degree")
        # revolute_joint_delta_min = min(revolute_joint_delta_min, np.rad2deg(joint_delta)) # 度

# 打印结果
print(f"Total successful (actual): {num_actual_valid[max_tries]} / {num_total_exp} = {num_actual_valid[max_tries] / num_total_exp:.2f}")
print(f"Total successful (pred): {num_pred_valid[max_tries]} / {num_total_exp} = {num_pred_valid[max_tries] / num_total_exp:.2f}")
print(f"Average number of triess: {num_total_tries} / {num_total_exp} = {num_total_tries / num_total_exp:.2f}")

print("jonit delta statistic (for pred as valid):")
print("prismatic:", prismatic_heap.get_min_k(5))
print("revolute:", revolute_heap.get_min_k(5))
# print("prismatic debug:", prismatic_heap_debug.get_min_k(5))
# print("prismatic:", prismatic_heap.numbers)
# print("revolute:", revolute_heap.numbers)

# 打印每个尝试次数的成功率
print("Detailed Results:")
for tries in range(1, max_tries + 1):
    success_rate_pred = num_pred_valid[tries] / num_total_exp
    success_rate_actual = num_actual_valid[tries] / num_total_exp
    print(f"tries={tries} | actual success rate: {success_rate_actual:.2f} | pred success rate {success_rate_pred:.2f}" )

# print("for valid explore, prismatic joint delta min: ", prismatic_joint_delta_min, "cm, revolute joint delta min: ", revolute_joint_delta_min, "degree")

# min_k = MinKNumbers()
# min_k.add_numbers([5, 2, 8, 3, 1, 9, 4, 6])
# print("当前所有数字:", min_k.numbers)  # 检查存储的数字是否正确
# print("最小的3个数字:", min_k.get_min_k(3))  # 检查输出