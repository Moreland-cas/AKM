#!/bin/bash

# 重建日志目录
LOG_DIR="$1"
# "/media/zby/MyBook/embody_analogy_data/assets/logs"

explore_run_name="$2"
# "explore_4_16"

recon_run_name="$3"
# "recon_4_16"
mkdir -p "$LOG_DIR/$recon_run_name"


#################### 超参在这里!! ####################
num_kframes=3
fine_lr=1e-3
save_memory=False
####################################################


# 遍历 LOG_DIR 下的文件夹
for obj_folder_path_explore in "$LOG_DIR/$explore_run_name"/*; do
    if [ -d "$obj_folder_path_explore" ]; then  # 确保是目录
        # 提取数据名称及其他参数
        obj_folder_name="$(basename "$obj_folder_path_explore")"
        
        obj_folder_path_reconstruct="$LOG_DIR/$recon_run_name/$obj_folder_name/"
        mkdir -p "$obj_folder_path_reconstruct"

        output_file="$obj_folder_path_reconstruct/output.txt"

        # 加一个逻辑, 如果已经有了则不再次执行
        if [ -f "$output_file" ] && [ "$(tail -n 1 "$output_file")" == "done" ]; then
            echo "Output file $output_file exists and the last line is 'done'. Skipping this reconstruction."
            continue
        fi

        # 执行 Python 脚本
        python /home/zby/Programs/Embodied_Analogy/scripts/test_reconstruct/test_reconstruct.py \
            --obj_folder_path_explore="$obj_folder_path_explore" \
            --obj_folder_path_reconstruct="$obj_folder_path_reconstruct" \
            --num_kframes=$num_kframes \
            --fine_lr=$fine_lr \
            --save_memory=$save_memory > "$output_file"  # 重定向输出
    fi
done

echo "所有命令已执行完成！"
