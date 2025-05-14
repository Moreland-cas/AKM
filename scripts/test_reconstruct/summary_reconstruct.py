import os
import argparse
import pickle
import numpy as np
from embodied_analogy.utility.constants import ASSET_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="recon_512", help='Folder where things are stored')
args = parser.parse_args()

root_path = os.path.join(ASSET_PATH, "logs", args.run_name)
results = []

# 遍历文件夹
for object_folder in os.listdir(root_path):
    object_path = os.path.join(root_path, object_folder)
    
    # with open(os.path.join(object_path, 'output.txt'), 'r') as output_file:
    #     output_lines = output_file.readlines()
    #     output = output_lines[-1]  # 获取最后一行
    #     assert output.strip() == "done"
    
    # 因为有些文件夹（没有成功 explore 的那些）没有result.pkl，所以需要try
    try:
        with open(os.path.join(object_path, 'result.pkl'), 'rb') as result_file:
            result = pickle.load(result_file)
            results.append(result)
    except Exception as e:
        print(f"Skip {object_path}: {e}, Since no result.pkl")
        continue  # 出现异常时跳过当前文件夹

def print_recon_result(results, prismatic_thresh, revolute_thresh):
    """
    统计误差在 prismatic_thresh (m) 和 revolute_thresh (degree) 范围内的重建数量, 以及定量误差
    """
    # print(f"Printing results under pris_thresh={prismatic_thresh} and rev_thresh={revolute_thresh}")
    # 统计有效结果
    num_exp = len(os.listdir(root_path))
    num_prismatic = 0
    num_prismatic_success = 0
    prismatic_angle_err_coarse = []
    prismatic_angle_err_fine = []

    num_revolute = 0
    num_revolute_success = 0
    revolute_pos_err_coarse = []
    revolute_angle_err_coarse = []
    revolute_pos_err_fine = []
    revolute_angle_err_fine = []

    type_loss = []

    # 同时明确一下失败的里面, 有几个是预测错误, 有几个是重建 error 太大判定为失败
    for i, result in enumerate(results):
        joint_type = result["gt_w"]["joint_type"]
        coarse_loss = result['coarse_loss']
        fine_loss = result['fine_loss']
        
        coarse_type_loss = coarse_loss['type_err']
        coarse_angle_err = coarse_loss['angle_err']
        coarse_pos_err = coarse_loss['pos_err']
        
        fine_type_loss = fine_loss['type_err']
        fine_angle_err = fine_loss['angle_err']
        fine_pos_err = fine_loss['pos_err']
        
        type_loss.append(fine_type_loss)
        
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
        
    # 打印结果
    print(f"Total successful exp: {len(results)} / {num_exp} = {len(results) / num_exp:.4f}")
    print(f"Total successful type classification: {len(results) - sum(type_loss)} / {len(results)} = {(len(results) - sum(type_loss)) / len(results):.4f}")

    print(f"Printing results under pris_thresh={prismatic_thresh} and rev_thresh={revolute_thresh}")
    
    print(f"Number of success prismatic: {num_prismatic_success}/{num_prismatic}, {num_prismatic_success/max(num_prismatic, 1):.4f}")
    prismatic_angle_err_coarse_mean = sum(prismatic_angle_err_coarse) / max(num_prismatic_success, 1)
    prismatic_angle_err_fine_mean = sum(prismatic_angle_err_fine) / max(num_prismatic_success, 1)
    print(f"\tcoarse angle loss: {np.rad2deg(prismatic_angle_err_coarse_mean):.4f} degree")
    print(f"\tfine   angle loss: {np.rad2deg(prismatic_angle_err_fine_mean):.4f} degree") 

    print(f"Number of success revolute: {num_revolute_success}/{num_revolute}, {num_revolute_success/max(num_revolute, 1):.4f}")
    revolute_angle_err_coarse_mean = sum(revolute_angle_err_coarse) / max(num_revolute_success, 1)
    revolute_angle_err_fine_mean = sum(revolute_angle_err_fine) / max(num_revolute_success, 1)
    print(f"\tcoarse angle loss: {np.rad2deg(revolute_angle_err_coarse_mean):.4f} degree")
    print(f"\tfine   angle loss: {np.rad2deg(revolute_angle_err_fine_mean):.4f} degree")

    revolute_pos_err_coarse_mean = sum(revolute_pos_err_coarse) / max(num_revolute_success, 1)
    revolute_pos_err_fine_mean = sum(revolute_pos_err_fine) / max(num_revolute_success, 1)
    print(f"\tcoarse pose loss: {revolute_pos_err_coarse_mean * 100:.4f} cm")
    print(f"\tfine   pose loss: {revolute_pos_err_fine_mean * 100:.4f} cm")
    

pris_thrs = [0.001, 0.005, 0.01, 0.01, 0.05, 0.1]
revo_thrs = [0.5, 1, 5, 10, 10, 20]
# 0.05 10 似乎是一个不错的
for i in range(len(pris_thrs)):
    print_recon_result(results, pris_thrs[i], revo_thrs[i])
    