#!/bin/bash

# 创建日志目录
LOG_DIR="/home/zby/Programs/Embodied_Analogy/assets/logs_complex"
run_name="test_explore"
mkdir -p "$LOG_DIR/$run_name"

test_data_cfg_path="/home/zby/Programs/Embodied_Analogy/scripts/test_data.json"

# 读取 JSON 文件中的所有键
test_data_cfgs=$(jq -r 'keys[]' "$test_data_cfg_path")

for test_data_cfg in $test_data_cfgs; do
    # 从 JSON 文件中提取相关信息
    joint_type=$(jq -r ".\"$test_data_cfg\".joint_type" "$test_data_cfg_path")
    asset_path=$(jq -r ".\"$test_data_cfg\".asset_path" "$test_data_cfg_path")
    instruction=$(jq -r ".\"$test_data_cfg\".instruction" "$test_data_cfg_path")
    obj_description=$(jq -r ".\"$test_data_cfg\".obj_description" "$test_data_cfg_path")
    obj_index=$(jq -r ".\"$test_data_cfg\".obj_index" "$test_data_cfg_path")
    joint_index=$(jq -r ".\"$test_data_cfg\".joint_index" "$test_data_cfg_path")
    load_pose=$(jq -r ".\"$test_data_cfg\".load_pose" "$test_data_cfg_path")
    load_quat=$(jq -r ".\"$test_data_cfg\".load_quat" "$test_data_cfg_path")
    load_scale=$(jq -r ".\"$test_data_cfg\".load_scale" "$test_data_cfg_path")
    active_link_name=$(jq -r ".\"$test_data_cfg\".active_link_name" "$test_data_cfg_path")
    active_joint_name=$(jq -r ".\"$test_data_cfg\".active_joint_name" "$test_data_cfg_path")

    # 提取数据名称及其他参数
    mkdir -p "$LOG_DIR/$run_name/${obj_index}_${joint_index}_${joint_type}/explore"
    output_file="${LOG_DIR}/$run_name/${obj_index}_${joint_index}_${joint_type}/explore/output.txt"

    # 执行 Python 脚本
    python /home/zby/Programs/Embodied_Analogy/scripts/test_explore.py \
        --logs_path="$LOG_DIR" \
        --run_name="$run_name" \
        --phy_timestep=0.004 \
        --planner_timestep=0.01 \
        --use_sapien2=True \
        --record_fps=30 \
        --pertubation_distance=0.1 \
        --valid_thresh=0.5 \
        --max_tries=5 \
        --update_sigma=0.05 \
        --reserved_distance=0.05 \
        --instruction="$instruction" \
        --obj_description="$obj_description" \
        --asset_path="$asset_path" \
        --joint_type="$joint_type" \
        --obj_index="$obj_index" \
        --joint_index="$joint_index" \
        --load_scale="$load_scale" \
        --load_pose="$load_pose" \
        --load_quat="$load_quat" \
        --active_link_name="$active_link_name" \
        --active_joint_name="$active_joint_name" > "$output_file"  # 重定向输出
    # TODO: 读取 save_dir 下的运行文件, 如果不是正常运行的话, 再次执行 python
    break
done

echo "所有命令已执行完成！"
