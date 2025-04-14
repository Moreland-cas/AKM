#!/bin/bash

# 重建日志目录
LOG_DIR="/home/zby/Programs/Embodied_Analogy/assets/logs"
run_name="4_14"

# 遍历 LOG_DIR 下的文件夹
for obj_folder in "$LOG_DIR/$run_name"/*; do
    if [ -d "$obj_folder" ]; then  # 确保是目录
        # 提取数据名称及其他参数
        mkdir -p "$LOG_DIR/$run_name/$(basename "$obj_folder")/reconstruct"
        output_file="$LOG_DIR/$run_name/$(basename "$obj_folder")/reconstruct/output.txt"

        # 执行 Python 脚本
        python /home/zby/Programs/Embodied_Analogy/scripts/test_reconstruct/test_reconstruct.py \
            --obj_folder="$obj_folder" \
            --num_kframes=5 \
            --fine_lr=1e-3 \
            --save_memory=False > "$output_file"  # 重定向输出
    fi
done

echo "所有命令已执行完成！"
