import argparse
import torch
import time
import random

def magic_computing(x, device='cuda'):
    """
    占用指定GB的显存,并进行一些低计算量的运算。
    
    参数:
        x (float): 占用的显存大小(以GB为单位)。
        device (str): 设备名称,默认为'cuda'。
    """
    # 将GB转换为字节
    bytes_to_occupy = int(x * 1e9)
    
    # 计算需要创建的张量大小(以字节为单位)
    # 假设使用float32类型,每个元素占用4字节
    num_elements = bytes_to_occupy // 4
    
    # 在指定设备上创建一个张量
    tensor = torch.randn(num_elements).to(device=device)
    
    # 执行一些低计算量的运算(例如加法)
    for i in range(100):
        # time.sleep(0.01)
        tensor = (tensor + 100) / 2.
    
    # 可以在这里添加一些日志输出,以便观察程序运行状态
    print(f"在设备 {device} 上占用 {x} GB 显存,进行低计算量运算...")

def main():
    # 在每张卡上动态占用显存
    while True:
        max_x = 7
        x = random.uniform(max_x / 2, max_x)  # 在 max_x / 2 到 max_x 之间随机取数
        magic_computing(x, device="cuda")
        time.sleep(0.1)  # 控制循环速度

if __name__ == "__main__":
    main()