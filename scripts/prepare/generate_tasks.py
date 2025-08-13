import os
import yaml
import argparse
import numpy as np

from akm.utility.constants import (
    PRISMATIC_JOINT_MAX_RANGE,
    REVOLUTE_JOINT_MAX_RANGE,
)
from akm.utility.utils import set_random_seed
from akm.utility.constants import ASSET_PATH, PROJECT_ROOT, SEED
from akm.utility.randomize_obj_pose import randomize_obj_load_pose

set_random_seed(SEED) # 666

"""
Returns a dict storing task_idx: task_info
task_info = {
    init_joint_state,
    manip_type,
    goal_delta,
    ...
}

For an object, tasks are divided into open, close, and different delta_states.
After determining these, determine init_joint_state and load_pose, a total of n random values.
"""
parser = argparse.ArgumentParser(description='Folder to save the cfg files')
parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, "cfgs", "task_cfgs_new"), help='folder to save the test cfg')
args = parser.parse_args()
    
task_cfgs = {}
obj_cfgs = []
# First traverse the prismatic and revolute folders to get information about all objects
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

# Here we get the complete obj_cfgs, then we need to traverse
global_task_idx = 0

# Based on the number of intervals in the final table and the max_range of the joint, 
# calculate multiple intervals and the number of tasks to be generated in each interval
prismatic_joint_range = (0, PRISMATIC_JOINT_MAX_RANGE)
revolute_joint_range = (0, REVOLUTE_JOINT_MAX_RANGE)

# Assume the final table is 4 * 4
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
    # NOTE Set load_joint_state to 0
    obj_cfg["obj_env_cfg"]["load_joint_state"] = 0
    
    # Run NUM_EXP_PER_SETTING times for each setting, and the load status will be different.
    obj_cfg = randomize_obj_load_pose(cfg=obj_cfg)
    obj_cfg["task_cfg"] = {}
    # NOTE The instruction in task_cfg is only used in the explore phase, so the default is "open"
    obj_cfg["task_cfg"]["instruction"] = "open the " + obj_cfg["obj_env_cfg"]["obj_description"]
    obj_cfg["task_cfg"]["task_id"] = global_task_idx
    obj_cfg["manip_env_cfg"] = {"tasks": {}}
    
    for start_grid_idx in range(num_grid):
        for end_grid_idx in range(num_grid):
            
            if start_grid_idx == end_grid_idx:
                continue
            
            # Determine manip_distance based on manip_type and (end_grid_idx - start_grid_idx)
            delta = pri_delta if (joint_type == "prismatic") else rev_delta
            manip_distance = (end_grid_idx - start_grid_idx) * delta
            
            # Randomly initialize manip_start_state and then calculate manip_end_state
            range_dict = prismatic_range_dict if (joint_type == "prismatic") else revolute_range_dict
            manip_start_state = np.random.uniform(low=range_dict[start_grid_idx][0], high=range_dict[start_grid_idx][1])
            manip_end_state = manip_start_state + manip_distance

            if joint_type == "revolute":
                # Since the distance of revolute is expressed in degrees, it needs to be converted to radians
                manip_start_state = np.deg2rad(manip_start_state)
                manip_end_state = np.deg2rad(manip_end_state)

            obj_cfg["manip_env_cfg"]["tasks"].update({
                f"{start_grid_idx}_{end_grid_idx}": {
                    "manip_start_state": float(manip_start_state),
                    "manip_end_state": float(manip_end_state)
                }
            })
                
    # Save the newly generated task_cfg
    task_cfgs[global_task_idx] = obj_cfg
    global_task_idx += 1

# Save as multiple yaml files
os.makedirs(args.save_dir, exist_ok=True)
for task_id, task_cfg in task_cfgs.items():
    yaml_file_path = os.path.join(args.save_dir, f"{task_id}.yaml")
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(task_cfg, f, default_flow_style=False, sort_keys=False)
print("Generate cfg files done.")
