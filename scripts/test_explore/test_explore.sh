#!/bin/bash

# 创建日志目录
LOG_DIR="$1"
# "/media/zby/MyBook/embody_analogy_data/assets/logs"
run_name="$2"
# "explore_4_16"
mkdir -p "$LOG_DIR/$run_name"

# test_data_cfg_path="/home/zby/Programs/Embodied_Analogy/scripts/test_data.json"
cfg_run_name="$3"
# "/home/zby/Programs/Embodied_Analogy/assets/logs/cfg_512"

#################### 超参在这里!! ####################
phy_timestep=0.004
planner_timestep=0.01
use_sapien2=True
record_fps=30
pertubation_distance=0.1
valid_thresh=0.5
max_tries=10
update_sigma=0.05
reserved_distance=0.05
num_initial_pts=1000
fully_zeroshot=False
use_anygrasp=False
offscreen=True
GPU_ID=7
####################################################
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 读取 JSON 文件中的所有键
# test_data_cfgs=$(jq -r 'keys[]' "$test_data_cfg_path")

for obj_folder_path_cfg in "$LOG_DIR/$cfg_run_name"/*; do
    # 提取数据名称及其他参数
    obj_folder_name="$(basename "$obj_folder_path_cfg")"
    obj_folder_path_explore="$LOG_DIR/$run_name/${obj_folder_name}"
    mkdir -p "$obj_folder_path_explore"
    output_file="${obj_folder_path_explore}/output.txt"

    # TODO: 首先读取 output_file，若 output_file 不存在或者 output_file 的最后一行不是 "done", 那么才跑底下的python 脚本，否则continue
    if [ -f "$output_file" ] && [ "$(tail -n 1 "$output_file")" == "done" ]; then
        echo "Output file $output_file exists and the last line is 'done'. Skipping this exploration."
        continue
    else
        echo "Output file $output_file does not exist or the last line is not 'done'. Running the Python script."
    fi

    # 执行 Python 脚本
    python test_explore.py \
        --obj_folder_path_cfg="$obj_folder_path_cfg" \
        --obj_folder_path_explore="$obj_folder_path_explore" \
        --phy_timestep="$phy_timestep" \
        --planner_timestep="$planner_timestep" \
        --use_sapien2="$use_sapien2" \
        --record_fps="$record_fps" \
        --pertubation_distance="$pertubation_distance" \
        --valid_thresh="$valid_thresh" \
        --fully_zeroshot="$fully_zeroshot" \
        --max_tries="$max_tries" \
        --update_sigma="$update_sigma" \
        --reserved_distance="$reserved_distance" \
        --num_initial_pts="$num_initial_pts" \
        --offscreen=$offscreen \
        --use_anygrasp="$use_anygrasp" > "$output_file"  
    break
done

echo "所有命令已执行完成！"
