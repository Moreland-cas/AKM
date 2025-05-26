# 与 summary
"""
打个表, 当然列可以改为具体的物体类

                        open                                close              
            scale1     scale2      scale3       scale1     scale2      scale3
prismatic
revolute 
"""

########################### 超参数 ###########################
prismatic_success_loss = 5 # cm
revolute_success_loss = 10 # degree
#############################################################

import os
import pickle
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help='Folder where things are stored')
parser.add_argument('--cfg_run_name', type=str, help='Folder where things are stored')
args = parser.parse_args()
from embodied_analogy.utility.constants import PROJECT_ROOT
root_path = os.path.join(PROJECT_ROOT, "assets/logs/", args.run_name)

summary_dict = {
    "prismatic": {
        # "open": {"scale": scale_dict}
        # scale_dict: {"losses": [], success: 10, failed: 20}
        "open": {}, 
        "close": {},
    },
    "revolute": {
        "open": {}, 
        "close": {},
    },
    # "max_tries": -1
}

def print_summary_dict(summary_dict):
    print("prismatic")
    print("\topen")
    for scale in summary_dict["prismatic"]["open"]:
        print(f"\t\tscale {scale}:")
        loss_list = summary_dict["prismatic"]["open"][scale]["loss_lists"]
        loss_list = [f"{loss:.2f}cm" for loss in loss_list]
        print(f"\t\t\t{loss_list}:")
    
    print("\tclose")
    for scale in summary_dict["prismatic"]["close"]:
        print(f"\t\tscale {scale}:")
        loss_list = summary_dict["prismatic"]["close"][scale]["loss_lists"]
        loss_list = [f"{loss:.2f}cm" for loss in loss_list]
        print(f"\t\t\t{loss_list}")

    print("revolute")
    print("\topen")
    for scale in summary_dict["revolute"]["open"]:
        print(f"\t\tscale {scale}:")
        loss_list = summary_dict["revolute"]["open"][scale]["loss_lists"]
        loss_list = [f"{loss:.2f}dg" for loss in loss_list]
        print(f"\t\t\t{loss_list}")

    print("\tclose")
    for scale in summary_dict["revolute"]["close"]:
        print(f"\t\tscale {scale}:")
        loss_list = summary_dict["revolute"]["close"][scale]["loss_lists"]
        loss_list = [f"{loss:.2f}dg" for loss in loss_list]
        print(f"\t\t\t{loss_list}")
    

def scaled_loss_success(loss, joint_type):
    final_loss = loss
    if joint_type == "prismatic":
        # return abs(final_loss) < 0.05
        return abs(final_loss) < prismatic_success_loss # cm
    else:
        # return abs(final_loss) < np.deg2rad(10)
        return abs(final_loss) < revolute_success_loss # degree
        
# 遍历文件夹
for object_folder in os.listdir(root_path):
    # /logs/manip_4_16/45135_1_prismatic
    object_path = os.path.join(root_path, object_folder)
    
    manip_types = ["close", "open"]
    for manip_type in manip_types:
        # /logs/manip_4_16/45135_1_prismatic/open
        manip_folder = os.path.join(object_path, manip_type)
        
        # 遍历不同 scale 的文件夹
        for scale_folder in os.listdir(manip_folder):
            # /logs/manip_4_16/45135_1_prismatic/open/scale_1
            scale_path = os.path.join(manip_folder, scale_folder)
            scale_value = float(scale_path.split("_")[-1])
            
            # 读取 output.txt 判断执行是否有效 不加这个了，因为有些实验连这个也没有
            # with open(os.path.join(scale_path, 'output.txt'), 'r') as output_file:
            #     output_lines = output_file.readlines()
            #     output = output_lines[-1]  # 获取最后一行
            #     assert output.strip() == "done"
    
            # 因为有些文件夹（没有成功 explore 的那些）没有result.pkl，所以需要try
            try:
            # if True:
                cfg_scale_path = scale_path.replace(args.run_name, args.cfg_run_name)
                with open(os.path.join(cfg_scale_path, "cfg.json"), 'r', encoding='utf-8') as file:
                    manip_cfg = json.load(file)
                    joint_type = manip_cfg["joint_type"]
                
                with open(os.path.join(scale_path, 'result.pkl'), 'rb') as result_file:
                    # 这个存储了多次操作后每次的 result
                    result = pickle.load(result_file)
                    
                if joint_type == "prismatic":
                    loss = result["diff"] * 100
                else:
                    loss = np.rad2deg(result["diff"])
                
                loss = abs(loss)
                
                if scale_value not in summary_dict[joint_type][manip_type]:
                    summary_dict[joint_type][manip_type][scale_value] = {
                        "loss_lists": [],
                        "success": 0,
                        "failed": 0
                    }
                else:
                    # 在这里取出 loss_list 的最后一个元素, 看一下是不是足够小
                    if scaled_loss_success(loss, joint_type):
                        summary_dict[joint_type][manip_type][scale_value]["loss_lists"].append(loss)
                        summary_dict[joint_type][manip_type][scale_value]["success"] += 1
                    else:
                        summary_dict[joint_type][manip_type][scale_value]["failed"] += 1
                # 从 cfg.json 中读取 max_tries
                # with open(os.path.join(scale_path, "cfg.json"), 'r', encoding='utf-8') as file:
                #     manip_cfg = json.load(file)
                #     summary_dict["max_tries"] = manip_cfg["max_manip"]
                
            except Exception as e:
                print(f"Skip {scale_path}: {e}, Since no result.pkl")
                continue  # 出现异常时跳过当前文件夹

# 统计有效结果
print_summary_dict(summary_dict)
# import sys;sys.exit()
# 对 summary_dict 进行处理, 得到一些统计量
def process_summary_dict(summary_dict):
    # max_tries = summary_dict["max_tries"]
    
    # NOTE: summary_dict 中存储的 loss 已经是 cm 或者 degree, 但是有正有负
    joint_types = ("prismatic", "revolute")
    manip_types = ("open", "close")
    
    for joint_type in joint_types:
        unit = "cm" if joint_type == "prismatic" else "degree"
        for manip_type in manip_types:
            for scale in summary_dict[joint_type][manip_type]:
                # print(f"{joint_type} {manip_type} {scale}")
                # num_manip = 0
                # 将所有 loss_list 补全到 max_tries 长度, 用原本数组的最后一个元素进行 pad
                loss_list = summary_dict[joint_type][manip_type][scale]["loss_lists"]
                loss_list = [abs(loss) for loss in loss_list]
                
                # num_exp * close_loop_times    
                # loss_array = np.array(summary_dict[joint_type][manip_type][scale]["loss_lists"]) 
                
                # if loss_array.ndim == 1:
                #     # 处理 num_exp == 1 的情况
                #     loss_array = loss_array.reshape(1, -1)
                
                print(f"\n{joint_type} {manip_type} {scale}")
                num_success = summary_dict[joint_type][manip_type][scale]["success"]
                num_failed = summary_dict[joint_type][manip_type][scale]["failed"]
                print(f"Success rate: {num_success / (num_success + num_failed) * 100}")
                # print(f"avg manip: {num_manip / loss_array.shape[0]}")
                if len(loss_list) > 0:
                    print(f"final error: {loss_list[-1]:.3f} {unit}")
                else:
                    print("Skip since len(loss_list) is 0.")
                # for i in range(max_tries):
                #     print(f"\tclose loop {i+1}: {loss_array[:, i].mean():.3f} {unit}")

process_summary_dict(summary_dict)
