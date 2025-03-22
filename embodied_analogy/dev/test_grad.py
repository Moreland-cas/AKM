import torch

# 假设 joint_dir 是一个需要梯度的张量
joint_dir = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")

skew_v = torch.zeros((3, 3), device="cuda")
skew_v[0, 1] = -joint_dir[2]
skew_v[0, 2] = joint_dir[1]
skew_v[1, 0] = joint_dir[2]
skew_v[1, 2] = -joint_dir[0]
skew_v[2, 0] = -joint_dir[1]
skew_v[2, 1] = joint_dir[0]
# skew_v.requires_grad_()


# 进行一些操作
output = (skew_v @ skew_v).sum()  # 计算输出

# 反向传播
output.backward()

# 查看 joint_dir 的梯度
print(joint_dir.grad)
