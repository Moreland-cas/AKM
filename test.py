import time
import torch

def occupy_gpu_memory(x, device='cuda'):
    """
    在一个无限循环中占用指定GB的显存，并进行一些低计算量的运算。
    
    参数:
        x (float): 占用的显存大小（以GB为单位）。
        device (str): 设备名称，默认为'cuda'。
    """
    # 将GB转换为字节
    bytes_to_occupy = int(x * 1e9)
    
    # 计算需要创建的张量大小（以字节为单位）
    # 假设使用float32类型，每个元素占用4字节
    num_elements = bytes_to_occupy // 4
    
    # 在指定设备上创建一个张量
    tensor = torch.randn(num_elements, device=device)
    
    # 无限循环
    while True:
        time.sleep(0.1)
        # 执行一些低计算量的运算（例如加法）
        tensor = (tensor + 100) / 2.
        
        # 可以在这里添加一些日志输出，以便观察程序运行状态
        # print(f"持续占用 {x} GB 显存，进行低计算量运算...")

# 示例：占用2GB显存
occupy_gpu_memory(7)