#!/bin/bash

# 定义数据根目录
DATA_ROOT="/home/zby/Programs/Embodied_Analogy/assets/dataset/"
PRISMATIC_DATA_ROOT="/home/zby/Programs/Embodied_Analogy/assets/dataset/one_drawer_cabinet"
REVOLUTE_DATA_ROOT="/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet"

# 定义实验参数
prismatic_sizes=(0.05 0.1 0.2)  # Prismatic 使用的尺寸
revolute_sizes=(0.2617993877991494 0.5235987755982988 0.7853981633974483)   # Revolute 使用的尺寸
revolute_sizes=(0.5235987755982988 0.7853981633974483)
actions=("open" "close")

# 创建日志目录
LOG_DIR="/home/zby/Programs/Embodied_Analogy/assets/logs"
mkdir -p "$LOG_DIR/prismatic"
mkdir -p "$LOG_DIR/revolute"

# # 处理 prismatic 数据
# for action in "${actions[@]}"; do
#     for size in "${prismatic_sizes[@]}"; do
#         # 创建对应的日志子目录
#         mkdir -p "$LOG_DIR/prismatic/$action/$size"

#         # 遍历 prismatic 数据目录
#         for prismatic_dir in "$PRISMATIC_DATA_ROOT"/*; do
#             if [ -d "$prismatic_dir" ]; then
#                 # 从目录名解析出必要的信息
#                 active_link_name="link_$(basename "$prismatic_dir" | awk -F'_' '{print $NF}')"
#                 active_joint_name="joint_$(basename "$prismatic_dir" | awk -F'_' '{print $NF}')"

#                 # 定义输出文件名
#                 data_name=$(basename "$prismatic_dir")
#                 output_file="${LOG_DIR}/prismatic/$action/$size/${data_name}.txt"

#                 # 执行 Python 脚本并将输出重定向到文件
#                 python test_one.py \
#                     --instruction="$action the cabinet" \
#                     --obj_description='cabinet' \
#                     --delta="$size" \
#                     --asset_path="$prismatic_dir" \
#                     --active_link_name="$active_link_name" \
#                     --active_joint_name="$active_joint_name" \
#                     --phy_timestep=0.004 \
#                     --planner_timestep=0.01 \
#                     --record_fps=30 \
#                     --pertubation_distance=0.1 \
#                     --valid_thresh=0.5 \
#                     --max_tries=10 \
#                     --update_sigma=0.05 \
#                     --num_initial_pts=1000 \
#                     --num_kframes=5 \
#                     --fine_lr=0.001 \
#                     --reloc_lr=0.003 \
#                     --reserved_distance=0.05 \
#                     --use_sapien2=True \
#                     --scale=1.0 > "$output_file"  # 重定向输出
#             fi
#         done
#     done
# done

# 处理 revolute 数据
for action in "${actions[@]}"; do
    for size in "${revolute_sizes[@]}"; do
        # 创建对应的日志子目录
        mkdir -p "$LOG_DIR/revolute/$action/$size"

        # 遍历 revolute 数据目录
        for revolute_dir in "$REVOLUTE_DATA_ROOT"/*; do
            if [ -d "$revolute_dir" ]; then
                # 从目录名解析出必要的信息
                active_link_name="link_$(basename "$revolute_dir" | awk -F'_' '{print $NF}')"
                active_joint_name="joint_$(basename "$revolute_dir" | awk -F'_' '{print $NF}')"

                # 定义输出文件名
                data_name=$(basename "$revolute_dir")
                output_file="${LOG_DIR}/revolute/$action/$size/${data_name}.txt"

                # 执行 Python 脚本并将输出重定向到文件
                python test_one.py \
                    --instruction="$action the cabinet" \
                    --obj_description='cabinet' \
                    --delta="$size" \
                    --asset_path="$revolute_dir" \
                    --active_link_name="$active_link_name" \
                    --active_joint_name="$active_joint_name" \
                    --phy_timestep=0.004 \
                    --planner_timestep=0.01 \
                    --record_fps=30 \
                    --pertubation_distance=0.1 \
                    --valid_thresh=0.5 \
                    --max_tries=5 \
                    --update_sigma=0.05 \
                    --num_initial_pts=1000 \
                    --num_kframes=5 \
                    --fine_lr=0.001 \
                    --reloc_lr=0.003 \
                    --reserved_distance=0.05 \
                    --use_sapien2=True \
                    --scale=1.0 > "$output_file"  # 重定向输出
            fi
        done
        exit 0
    done
done

echo "所有命令已执行完成！"
