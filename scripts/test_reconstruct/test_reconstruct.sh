#!/bin/bash

# 重建日志目录
LOG_DIR="$1"
# "/media/zby/MyBook/embody_analogy_data/assets/logs"

explore_run_name="$2"
# "explore_4_16"

recon_run_name="$3"
# "recon_4_16"

mkdir -p "$LOG_DIR/$recon_run_name"

# 遍历 LOG_DIR 下的文件夹
for obj_folder_path_explore in "$LOG_DIR/$explore_run_name"/*; do
    if [ -d "$obj_folder_path_explore" ]; then  # 确保是目录
        # 提取数据名称及其他参数
        obj_folder_name="$(basename "$obj_folder_path_explore")"
        
        obj_folder_path_reconstruct="$LOG_DIR/$recon_run_name/$obj_folder_name/"
        mkdir -p "$obj_folder_path_reconstruct"

        output_file="$obj_folder_path_reconstruct/output.txt"

        # 执行 Python 脚本
        python /home/zby/Programs/Embodied_Analogy/scripts/test_reconstruct/test_reconstruct.py \
            --obj_folder_path_explore="$obj_folder_path_explore" \
            --obj_folder_path_reconstruct="$obj_folder_path_reconstruct" \
            --num_kframes=5 \
            --fine_lr=1e-3 \
            --save_memory=False > "$output_file"  # 重定向输出
    fi
    break
done

echo "所有命令已执行完成！"
