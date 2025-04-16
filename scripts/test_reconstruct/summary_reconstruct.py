import os
import pickle
import numpy as np

root_path = "/media/zby/MyBook/embody_analogy_data/assets/logs/recon_4_16"
results = []

# 遍历文件夹
for object_folder in os.listdir(root_path):
    object_path = os.path.join(root_path, object_folder)
    
    with open(os.path.join(object_path, 'output.txt'), 'r') as output_file:
        output_lines = output_file.readlines()
        output = output_lines[-1]  # 获取最后一行
        assert output.strip() == "done"
    
    # 因为有些文件夹（没有成功 explore 的那些）没有result.pkl，所以需要try
    try:
        with open(os.path.join(object_path, 'result.pkl'), 'rb') as result_file:
            result = pickle.load(result_file)
            results.append(result)
    except Exception as e:
        print(f"Skip {object_path}: {e}, Since no result.pkl")
        continue  # 出现异常时跳过当前文件夹

# 统计有效结果
num_exp = len(os.listdir(root_path))
num_prismatic = 0
prismatic_angle_err_coarse = []
prismatic_angle_err_fine = []

num_revolute = 0
revolute_pos_err_coarse = []
revolute_angle_err_coarse = []
revolute_pos_err_fine = []
revolute_angle_err_fine = []

# 打印所有 prismatic 的平均
# 打印所有 revolute 的平均
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
    
    if joint_type == "prismatic":
        num_prismatic += 1
        prismatic_angle_err_coarse.append(coarse_angle_err)
        prismatic_angle_err_fine.append(fine_angle_err)
    else:
        num_revolute += 1
        revolute_pos_err_coarse.append(coarse_pos_err)
        revolute_angle_err_coarse.append(coarse_angle_err)
        revolute_pos_err_fine.append(fine_pos_err)
        revolute_angle_err_fine.append(fine_angle_err)
        

# 打印结果
print(f"Total successful exp: {len(results)} / {num_exp} = {len(results) / num_exp:.4f}")
print(f"Number of prismatic: {num_prismatic}")
prismatic_angle_err_coarse_mean = sum(prismatic_angle_err_coarse) / num_prismatic
prismatic_angle_err_fine_mean = sum(prismatic_angle_err_fine) / num_prismatic
print(f"\tcoarse angle loss: {np.rad2deg(prismatic_angle_err_coarse_mean):.4f} degree")
print(f"\tfine   angle loss: {np.rad2deg(prismatic_angle_err_fine_mean):.4f} degree") 

print(f"Number of revolute: {num_revolute}")
revolute_angle_err_coarse_mean = sum(revolute_angle_err_coarse) / num_revolute
revolute_angle_err_fine_mean = sum(revolute_angle_err_fine) / num_revolute
print(f"\tcoarse angle loss: {np.rad2deg(revolute_angle_err_coarse_mean):.4f} degree")
print(f"\tfine   angle loss: {np.rad2deg(revolute_angle_err_fine_mean):.4f} degree")

revolute_pos_err_coarse_mean = sum(revolute_pos_err_coarse) / num_revolute
revolute_pos_err_fine_mean = sum(revolute_pos_err_fine) / num_revolute
print(f"\tcoarse pose loss: {revolute_pos_err_coarse_mean * 100:.4f} cm")
print(f"\tfine   pose loss: {revolute_pos_err_fine_mean * 100:.4f} cm")