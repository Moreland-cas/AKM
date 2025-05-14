import subprocess
import math
import sys

def run_task_on_gpu(gpu_id, task_id):
    # 设置环境变量，指定当前任务使用的GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 调用test函数的Python脚本，传递任务ID和GPU索引
    command = ["python", "test_script.py", str(task_id), str(gpu_id)]
    subprocess.run(command, env=env, check=True)

def main():
    total_tasks = 100  # 总任务数
    visible_devices = sys.argv[1]  # 从命令行参数获取CUDA_VISIBLE_DEVICES
    gpu_ids = list(map(int, visible_devices.split(',')))  # 将设备索引转换为整数列表
    num_gpus = len(gpu_ids)  # 可用的GPU数量

    print(f"Total tasks: {total_tasks}")
    print(f"Visible GPUs: {visible_devices}")

    # 计算每个GPU需要处理的任务数量
    tasks_per_gpu = math.ceil(total_tasks / num_gpus)

    # 为每个GPU分配任务并运行
    for gpu_index, gpu_id in enumerate(gpu_ids):
        start_task = gpu_index * tasks_per_gpu
        end_task = min(start_task + tasks_per_gpu, total_tasks)

        print(f"GPU {gpu_id}: Handling tasks {start_task} to {end_task - 1}")
        for task_id in range(start_task, end_task):
            print(f"Running task {task_id} on GPU {gpu_id}")
            run_task_on_gpu(gpu_id, task_id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <CUDA_VISIBLE_DEVICES>")
        sys.exit(1)
    main()
    

# test.py
import sys
import os

def test(task_id, gpu_id):
    print(f"Running task {task_id} on GPU {gpu_id}")
    # 在这里实现你的任务逻辑
    # 例如：执行一些计算任务

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_script.py <task_id> <gpu_id>")
        sys.exit(1)
    task_id = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    test(task_id, gpu_id)