import torch

def tracks3d_variance(tracks: torch.Tensor) -> torch.Tensor:
    """
    计算 T, M, 3 形状的 3D 轨迹数据的平均方差 (PyTorch 版本)。

    参数:
        tracks (torch.Tensor): 形状为 (T, M, 3) 的张量，表示 T 个时刻 M 个点的 3D 轨迹。

    返回:
        torch.Tensor: M 个点的方差均值（标量）。
    """
    # 计算每个点在 T 个时刻的方差 (M, 3)
    pointwise_variance = torch.var(tracks, dim=0, unbiased=False)  # 计算 M 个点的 3 维方差
    
    # 计算每个点的总方差（对 3 维坐标求均值）
    pointwise_variance_mean = torch.mean(pointwise_variance, dim=1)  # (M,)

    # 计算所有点的方差均值
    average_variance = torch.mean(pointwise_variance_mean)

    return average_variance

# 示例数据
T, M = 100, 10
tracks1 = torch.rand(T, M, 3)  # 生成随机 3D 轨迹 (T, M, 3)
tracks2 = torch.rand(T, M, 3) + torch.arange(T).unsqueeze(-1).unsqueeze(-1)  # 生成随机 3D 轨迹 (T, M, 3)

# 计算平均方差
avg_var = tracks3d_variance(tracks1)
print(f"Average Variance: {avg_var.item()}")

avg_var = tracks3d_variance(tracks2)
print(f"Average Variance: {avg_var.item()}")