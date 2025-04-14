import os
import pickle
import numpy as np

root_path = "/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore_4_11"
# root_path = "/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore"
results = []

# 遍历文件夹
for object_folder in os.listdir(root_path):
    object_path = os.path.join(root_path, object_folder, "explore")
    try:
        # 读取 cfg.json
        with open(os.path.join(object_path, 'cfg.json'), 'r') as cfg_file:
            cfg = cfg_file.read()  # 根据需要解析 JSON

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

# 统计有效结果
num_valid = 0
num_exp = len(results)
total_tries = 0
attempts_count = {i: 0 for i in range(1, 11)}  # 尝试次数从 1 到 10 的统计

prismatic_joint_delta_min = 1e6
revolute_joint_delta_min = 1e6

for i, result in enumerate(results):
    has_valid_explore = result["has_valid_explore"]
    
    if has_valid_explore:
        num_valid += 1
    else:
        continue
    
    num_tries = result["num_tries"]
    total_tries += num_tries
    
    if num_tries <= 10:  # 只统计尝试次数在 1 到 10 的情况
        for i in range(num_tries, 11):
            attempts_count[i] += 1
            
    # 看一下成功的里面的打开角度是多少
    joint_type = result["joint_type"]
    joint_delta = result["joint_state_end"] - result["joint_state_start"]
    if joint_type == "prismatic":
        prismatic_joint_delta_min = min(prismatic_joint_delta_min, joint_delta * 100) # cm
        # print(joint_delta * 100, "cm")
    else:
        print(np.rad2deg(joint_delta), "degree")
        revolute_joint_delta_min = min(revolute_joint_delta_min, np.rad2deg(joint_delta)) # 度
        

# 打印结果
print(f"Total successful attempts: {num_valid} / {num_exp} = {num_valid / num_exp:.2f}")
print(f"Average number of tries for successful attempts: {total_tries} / {num_valid} = {total_tries / num_valid:.2f}")

# 打印每个尝试次数的成功率
for tries in range(1, 11):
    success_rate = attempts_count[tries] / num_exp if num_exp > 0 else 0
    print(f"Success rate for attempts <= {tries}: {success_rate:.2f}")

print("for valid explore, prismatic joint delta min: ", prismatic_joint_delta_min, "cm, revolute joint delta min: ", revolute_joint_delta_min, "degree")