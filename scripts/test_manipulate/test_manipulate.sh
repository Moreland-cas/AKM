#!/bin/bash

# 重建日志目录
LOG_DIR="$1"
# "/media/zby/MyBook/embody_analogy_data/assets/logs"

recon_run_name="$2"
# "recon_4_16"

manip_run_name="$3"
# "manip_4_11"
mkdir -p "$LOG_DIR/$manip_run_name"

#################### 超参在这里!! ####################
            # easy  subset
prismatic_max_distance=0.3
revolute_max_distance=35
prismatic_manip_distances=(0.1 0.15 0.2)
revolute_manip_distances=(10 20 30)

reloc_lr=3e-3
whole_traj_close_loop=True
# drop_large_move=True
prismatic_recon_success_thresh=0.05
revolute_recon_success_thresh=10

# used for whole-traj close-loop
max_manip=5
prismatic_whole_traj_success_thresh=0.01 # m
revolute_whole_traj_success_thresh=5 # degree

# only used for inter close-loop
max_attempts=5 # 等于1时代表没有 close_loop
prismatic_reloc_interval=0.05 # m
prismatic_reloc_tolerance=0.01 # m
revolute_reloc_interval=5  # degree
revolute_reloc_tolerance=2 # degree
####################################################

# 遍历 LOG_DIR 下的重建文件夹
for obj_folder_path_reconstruct in "$LOG_DIR/$recon_run_name"/*; do
    if [ -d "$obj_folder_path_reconstruct" ]; then  # 确保是目录
        # 提取数据名称
        obj_folder_name="$(basename "$obj_folder_path_reconstruct")"

        for operation in open close; do
            operation_dir="$LOG_DIR/$manip_run_name/$obj_folder_name/$operation"
            mkdir -p "$operation_dir"

            # 根据 obj_folder_name 判断 joint_type
            # 如果是 prismatic, 则把 manipulate distances 设置为 prismatic_manip_distances
            if [[ $obj_folder_name == *"prismatic"* ]]; then
                manipulate_distances=("${prismatic_manip_distances[@]}")
                max_distance=$prismatic_max_distance
            else
                manipulate_distances=("${revolute_manip_distances[@]}")
                max_distance=$revolute_max_distance
            fi

            # 遍历 manipulate distances
            for distance in "${manipulate_distances[@]}"; do
                scale_dir="$operation_dir/scale_$distance"
                mkdir -p "$scale_dir"

                output_file="$scale_dir/output.txt"
                
                # 加一个逻辑, 如果已经有了则不再次执行
                if [ -f "$output_file" ] && [ "$(tail -n 1 "$output_file")" == "done" ]; then
                    # echo "Output file $output_file exists and the last line is 'done'. Skipping this manipulation."
                    continue
                fi
                echo "Executing Python script for $output_file"
                echo $obj_folder_path_reconstruct
                # 执行 Python 脚本
                python /home/zby/Programs/Embodied_Analogy/scripts/test_manipulate/test_manipulate.py \
                    --obj_folder_path_reconstruct="$obj_folder_path_reconstruct" \
                    --scale_dir="$scale_dir" \
                    --manipulate_type="$operation" \
                    --manipulate_distance="$distance" \
                    --whole_traj_close_loop=$whole_traj_close_loop \
                    # --drop_large_move=$drop_large_move \
                    --reloc_lr=$reloc_lr \
                    --max_manip=$max_manip \
                    --prismatic_whole_traj_success_thresh=$prismatic_whole_traj_success_thresh \
                    --revolute_whole_traj_success_thresh=$revolute_whole_traj_success_thresh \
                    --max_attempts=$max_attempts \
                    --prismatic_reloc_interval=$prismatic_reloc_interval \
                    --prismatic_reloc_tolerance=$prismatic_reloc_tolerance \
                    --revolute_reloc_interval=$revolute_reloc_interval \
                    --revolute_reloc_tolerance=$revolute_reloc_tolerance \
                    --prismatic_recon_success_thresh=$prismatic_recon_success_thresh \
                    --revolute_recon_success_thresh=$revolute_recon_success_thresh \
                    --max_distance="$max_distance" > "$output_file"  # 重定向输出
            done
        done
    fi
done

echo "所有命令已执行完成！"
