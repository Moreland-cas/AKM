import torch
import numpy as np

def torch_minimize(init_params, loss_fn, lr=0.01, max_iters=1000, tol=1e-6):
    # 转换 numpy 数组为 PyTorch 张量
    init_params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    
    # 选择优化器
    optimizer = torch.optim.Adam([init_params], lr=lr)
    
    prev_loss = float('inf')
    for i in range(max_iters):
        optimizer.zero_grad()
        loss = loss_fn(init_params)
        loss.backward()
        optimizer.step()
        
        # 计算损失变化
        loss_change = abs(loss.item() - prev_loss)
        if loss_change < tol:
            break
        prev_loss = loss.item()
    
    # return {
    #     'x': init_params.detach().numpy(),  # 优化后的参数转换为 numpy
    #     'fun': prev_loss,  # 最优损失值
    #     'nit': i+1,  # 迭代次数
    #     'success': loss_change < tol
    # }
    optimized_params = init_params.detach().numpy()
    return optimized_params


# 示例使用
if __name__ == "__main__":
    # 目标函数: (x-3)^2 最小化
    def loss_fn(x):
        return (x - 3) ** 2
    
    x_init = np.array([0.0])  # 初始值（numpy 数组）
    result = torch_minimize(x_init, loss_fn, lr=0.1, max_iters=1000, tol=1e-6, optimizer_type='adam')
    print(result)
