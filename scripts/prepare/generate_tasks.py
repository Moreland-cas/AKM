import os
import copy
import json
import numpy as np
import argparse
from embodied_analogy.utility.constants import ASSET_PATH, PROJECT_ROOT, SEED
from embodied_analogy.utility.randomize_obj_pose import randomize_obj_load_pose, randomize_dof
from embodied_analogy.utility.utils import set_random_seed
from embodied_analogy.utility.constants import (
    NUM_EXP_PER_SETTING,
    PRISMATIC_JOINT_MAX_RANGE,
    PRISMATIC_TEST_JOINT_DELTAS,
    REVOLUTE_JOINT_MAX_RANGE,
    REVOLUTE_TEST_JOINT_DELTAS
)
set_random_seed(SEED)

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
parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, "scripts"), help='folder to save the test cfg')
args = parser.parse_args()
    
task_cfgs = {}
obj_cfgs = []

pri_path = os.path.join(ASSET_PATH, "dataset/one_drawer_cabinet")
# 44781_link_0
for tmp_folder in os.listdir(pri_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        "joint_type": "prismatic",
        "data_path": os.path.join("dataset/one_drawer_cabinet", tmp_folder),
        "obj_index": obj_index,
        "joint_index": joint_index,
        "obj_description": "cabinet",
        "load_pose": None,
        "load_quat": None,
        "load_scale": None,
        "active_link_name": f"link_{joint_index}",
        "active_joint_name": f"joint_{joint_index}",
        "instruction": None,
        "init_joint_state": None
    }
    obj_cfgs.append(tmp_dict)

rev_path = os.path.join(ASSET_PATH, "dataset/one_door_cabinet")
for tmp_folder in os.listdir(rev_path):
    joint_index = tmp_folder.split("_")[-1]
    obj_index = tmp_folder.split("_")[0]
    tmp_dict = {
        "joint_type": "revolute",
        "data_path": os.path.join("dataset/one_door_cabinet", tmp_folder),
        "obj_index": obj_index,
        "joint_index": joint_index,
        "obj_description": "cabinet",
        "load_pose": None,
        "load_quat": None,
        "load_scale": None,
        "active_link_name": f"link_{joint_index}",
        "active_joint_name": f"joint_{joint_index}",
        "instruction": None,
        "init_joint_state": None
    }
    obj_cfgs.append(tmp_dict)

# 这里得到了完整的 obj_cfgs, 接下来需要遍历
global_task_idx = 0
for obj_cfg in obj_cfgs:
    joint_type = obj_cfg["joint_type"]
    # 根据 joint_type 确定 manip 的区间大小和 obj_init_dof_low 和 obj_init_dof_high
    if joint_type == "prismatic":
        max_joint_range = PRISMATIC_JOINT_MAX_RANGE
        test_joint_deltas = PRISMATIC_TEST_JOINT_DELTAS
    elif joint_type == "revolute":
        max_joint_range = REVOLUTE_JOINT_MAX_RANGE
        test_joint_deltas = REVOLUTE_TEST_JOINT_DELTAS
    
    for manip_type in ["open", "close"]:
        for test_delta in test_joint_deltas:
            assert max_joint_range >= test_delta
            if manip_type == "open":
                obj_init_dof_low = 0
                obj_init_dof_high = max_joint_range - test_delta
            else:
                obj_init_dof_low = test_delta
                obj_init_dof_high = max_joint_range

            if joint_type == "revolute":
                # 由于 revolute 的 distance 是用 degree 表示的, 因此需要转换为弧度
                obj_init_dof_low = np.deg2rad(obj_init_dof_low)
                obj_init_dof_high = np.deg2rad(obj_init_dof_high)

            for _ in range(NUM_EXP_PER_SETTING):
                base_obj_cfg = copy.copy(obj_cfg)
                base_obj_cfg = randomize_obj_load_pose(
                    cfg=base_obj_cfg,
                    dof_low=obj_init_dof_low,
                    dof_high=obj_init_dof_high
                )
                base_obj_cfg["instruction"] = manip_type + " the " + base_obj_cfg["obj_description"]
                # 将新产生的 task_cfg 进行保存
                task_cfgs[global_task_idx] = base_obj_cfg
                global_task_idx += 1

# print(task_cfgs)
# for k, v in task_cfgs.items():
#     print(k)
#     print(v)
    
os.makedirs(args.save_dir, exist_ok=True)
cfg_file_path = os.path.join(args.save_dir, "test_cfgs.json")
with open(cfg_file_path, 'w', encoding='utf-8') as f:
    json.dump(task_cfgs, f, ensure_ascii=False, indent=4)
print("Generate cfg files done.")
