import yaml
import os
import copy
import numpy as np
import argparse
from akm.utility.constants import ASSET_PATH, PROJECT_ROOT, SEED
from akm.utility.randomize_obj_pose import randomize_obj_load_pose
from akm.utility.utils import set_random_seed
from akm.utility.constants import (
    # NUM_EXP_PER_SETTING,
    PRISMATIC_JOINT_MAX_RANGE,
    # PRISMATIC_TEST_JOINT_DELTAS,
    REVOLUTE_JOINT_MAX_RANGE,
    # REVOLUTE_TEST_JOINT_DELTAS
)
set_random_seed(SEED) # 666

"""
返回一个 dict, 存储 task_idx: task_info
    task_info = {
        init_joint_state,
        manip_type,
        goal_delta,
        ...
    }

对于一个物体, task 分为 open, close, 和不同 delta_state
确定好这些后, 再确定 init_joint_state 和 load_pose, 一共随机 n 个
"""
parser = argparse.ArgumentParser(description='Folder to save the cfg files')
parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, "cfgs", "task_cfgs_new"), help='folder to save the test cfg')
args = parser.parse_args()
    
task_cfgs = {}
obj_cfgs = []
# 首先遍历 prismatic 和 revolute folder, 得到所有 object 的信息
pri_path = os.path.join(ASSET_PATH, "dataset/one_drawer_cabinet")
# 44781_link_0
for tmp_folder in os.listdir(pri_path):
    joint_index = int(tmp_folder.split("_")[-1])
    obj_index = int(tmp_folder.split("_")[0])
    tmp_dict = {
        "obj_env_cfg": {
            "joint_type": "prismatic",
            "data_path": os.path.join("dataset/one_drawer_cabinet", tmp_folder),
            "obj_index": obj_index,
            "joint_index": joint_index,
            "obj_description": "cabinet",
            "active_link_name": f"link_{joint_index}",
            "active_joint_name": f"joint_{joint_index}",
        }
    }
    obj_cfgs.append(tmp_dict)

rev_path = os.path.join(ASSET_PATH, "dataset/one_door_cabinet")
for tmp_folder in os.listdir(rev_path):
    joint_index = int(tmp_folder.split("_")[-1])
    obj_index = int(tmp_folder.split("_")[0])
    tmp_dict = {
        "obj_env_cfg": {
            "joint_type": "revolute",
            "data_path": os.path.join("dataset/one_door_cabinet", tmp_folder),
            "obj_index": obj_index,
            "joint_index": joint_index,
            "obj_description": "cabinet",
            "active_link_name": f"link_{joint_index}",
            "active_joint_name": f"joint_{joint_index}"
        }
    }
    obj_cfgs.append(tmp_dict)

# 这里得到了完整的 obj_cfgs, 接下来需要遍历
global_task_idx = 0

# 根据最终表格的区间个数, 和关节的 max_range, 计算多个区间，以及每个区间要生成的任务个数
prismatic_joint_range = (0, PRISMATIC_JOINT_MAX_RANGE)
revolute_joint_range = (0, REVOLUTE_JOINT_MAX_RANGE)

# 假设最后的表格是 4 * 4 的
num_grid = 4
prismatic_range_dict = {}
revolute_range_dict = {}

pri_delta = PRISMATIC_JOINT_MAX_RANGE / num_grid # 10 cm
rev_delta = REVOLUTE_JOINT_MAX_RANGE / num_grid # 10 degree
    
for i in range(num_grid):
    prismatic_range_dict[i] = (pri_delta * i, pri_delta * (i+1))
    revolute_range_dict[i] = (rev_delta * i, rev_delta * (i+1))

for obj_cfg in obj_cfgs:
    joint_type = obj_cfg["obj_env_cfg"]["joint_type"]
    # NOTE 将 load_joint_state 设置为 0
    obj_cfg["obj_env_cfg"]["load_joint_state"] = 0
    
    # 每一种 setting 下运行 NUM_EXP_PER_SETTING 遍, 且 load 状态会有所不同
    # base_obj_cfg = copy.deepcopy(obj_cfg)
    obj_cfg = randomize_obj_load_pose(cfg=obj_cfg)
    obj_cfg["task_cfg"] = {}
    # NOTE 这里 task_cfg 中的 instruction 仅用在 explore 阶段, 因此默认用 "open"
    obj_cfg["task_cfg"]["instruction"] = "open the " + obj_cfg["obj_env_cfg"]["obj_description"]
    obj_cfg["task_cfg"]["task_id"] = global_task_idx
    obj_cfg["manip_env_cfg"] = {"tasks": {}}
    
    for start_grid_idx in range(num_grid):
        for end_grid_idx in range(num_grid):
            
            if start_grid_idx == end_grid_idx:
                continue
            
            # 根据 manip_type 和 (end_grid_idx - start_grid_idx) 确定 manip_distance
            delta = pri_delta if (joint_type == "prismatic") else rev_delta
            manip_distance = (end_grid_idx - start_grid_idx) * delta
            
            # 随机初始化 manip_start_state, 并随之计算 manip_end_state
            range_dict = prismatic_range_dict if (joint_type == "prismatic") else revolute_range_dict
            manip_start_state = np.random.uniform(low=range_dict[start_grid_idx][0], high=range_dict[start_grid_idx][1])
            manip_end_state = manip_start_state + manip_distance

            if joint_type == "revolute":
                # 由于 revolute 的 distance 是用 degree 表示的, 因此需要转换为弧度
                manip_start_state = np.deg2rad(manip_start_state)
                manip_end_state = np.deg2rad(manip_end_state)

            obj_cfg["manip_env_cfg"]["tasks"].update({
                f"{start_grid_idx}_{end_grid_idx}": {
                # (start_grid_idx, end_grid_idx): {
                    "manip_start_state": float(manip_start_state),
                    "manip_end_state": float(manip_end_state)
                }
            })
                
    # 将新产生的 task_cfg 进行保存
    task_cfgs[global_task_idx] = obj_cfg
    global_task_idx += 1

# print(task_cfgs)
# for k, v in task_cfgs.items():
#     print(k)
#     print(v)

# 保存为 .json
# os.makedirs(args.save_dir, exist_ok=True)
# cfg_file_path = os.path.join(args.save_dir, "test_cfgs.json")
# with open(cfg_file_path, 'w', encoding='utf-8') as f:
#     json.dump(task_cfgs, f, ensure_ascii=False, indent=4)
# print("Generate cfg files done.")

# 保存为多个 yaml 文件
os.makedirs(args.save_dir, exist_ok=True)
for task_id, task_cfg in task_cfgs.items():
    # if task_id == 60:
    #     pass
    yaml_file_path = os.path.join(args.save_dir, f"{task_id}.yaml")
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(task_cfg, f, default_flow_style=False, sort_keys=False)
        # yaml.dump(task_cfg, f)
print("Generate cfg files done.")
