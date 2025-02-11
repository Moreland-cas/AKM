import torch
import numpy as np
import time

def test_conversion_time():
    # 设定数组大小
    height, width = 600, 800
    
    # 创建一个随机的 numpy 数组
    np_array = np.random.rand(height, width)
    
    # 记录转换前的时间
    start_time = time.time()
    
    # 将 numpy 数组转换为 torch 张量，并转移到 GPU（假设 GPU 可用）
    tensor = torch.tensor(np_array, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(np_array, dtype=torch.float32)
    
    # 记录转换后的时间
    end_time = time.time()
    
    # 计算并打印转换所花费的时间
    print(f"Time taken to convert numpy array to torch tensor and move to GPU (if available): {end_time - start_time:.6f} seconds")
    
# 调用函数进行测试
test_conversion_time()
