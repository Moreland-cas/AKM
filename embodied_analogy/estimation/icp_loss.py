import torch
import numpy as np
from embodied_analogy.estimation.utils import *

def icp_loss_torch(
    joint_axis_scaled, 
    ref_pc, tgt_pc, 
    target_normals=None, 
    loss_type="point_to_point", 
    joint_type="prismatic",
    icp_select_range=0.03
):
    """
    计算 ICP 损失 (点到点 或 点到平面)，支持平移或旋转。
    需要优化的是 joint_axis_scaled, joint_axis_scaled 是唯一需要梯度的

    参数：
        joint_axis_scaled (torch.Tensor): 形状 (3,) ，是可学习的参数
        ref_pc (np.ndarray): 参考点云，形状 (N, 3)
        tgt_pc (np.ndarray): 目标点云，形状 (M, 3)
        target_normals (np.ndarray, 可选): 目标点云法向量，形状 (M, 3)，仅在点到平面模式中使用
        loss_type (str): "point_to_point" 或 "point_to_plane" # TODO: 这里还可以加一种混合式
        joint_type (str): "prismatic" 或 "revolute"

    返回：
        loss (torch.Tensor): 计算得到的损失值
    """
    assert joint_axis_scaled.is_cuda and "joint axis must be on GPU"
    assert joint_axis_scaled.requires_grad and "joint axis must requires grad to be optimized"
    assert loss_type in ["point_to_point", "point_to_plane"]
    assert joint_type in ["prismatic", "revolute"]
    # print(f"len(ref_pc)={len(ref_pc)} len(tgt_pc)={len(tgt_pc)}")
    
    if min(len(ref_pc), len(tgt_pc)) < 100:
        return torch.tensor(0).cuda()

    # 将 numpy 数组转换为 torch.Tensor
    ref_pc = torch.tensor(ref_pc, dtype=torch.float32, device=joint_axis_scaled.device)
    target_pc = torch.tensor(tgt_pc, dtype=torch.float32, device=joint_axis_scaled.device)

    if loss_type == "point_to_plane":
        if target_normals is None:
            target_normals = compute_normals(tgt_pc) # N, 3
        target_normals = torch.tensor(target_normals, dtype=torch.float32, device=joint_axis_scaled.device)

    # 初始化变换矩阵 Tref2tgt (4x4)
    Tref2tgt = torch.eye(4, device=ref_pc.device)

    if joint_type == "prismatic":
        # 平移变换: joint_axis_scaled 作为平移向量
        Tref2tgt[:3, 3] = joint_axis_scaled
    elif joint_type == "revolute":
        # 旋转变换: 使用 joint_axis_scaled 作为旋转轴，构造旋转矩阵
        axis = joint_axis_scaled / torch.norm(joint_axis_scaled)  # 归一化旋转轴
        theta = torch.norm(joint_axis_scaled)  # 假设角度为旋转轴的模
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=ref_pc.device)
        R = torch.eye(3, device=ref_pc.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        Tref2tgt[:3, :3] = R

    # 将 ref_pc 扩展到齐次坐标 (N, 4)
    ref_pc_homogeneous = torch.cat([ref_pc, torch.ones(ref_pc.shape[0], 1, device=ref_pc.device)], dim=1)

    # 应用变换
    transformed_ref_pc_homogeneous = ref_pc_homogeneous @ Tref2tgt.T  # 变换为齐次坐标 (N, 4)
    transformed_ref_pc = transformed_ref_pc_homogeneous[:, :3]  # 去掉齐次坐标 (N, 3)

    # 最近邻搜索
    # TODO: 处理找不到最近点的情况
    indices, valid_mask = find_correspondences(transformed_ref_pc.detach().cpu().numpy(), tgt_pc, max_distance=icp_select_range)
    
    if valid_mask.sum() < 100: # 如果太少返回 0
        return torch.tensor(0).cuda()

    matched_target_pc = target_pc[indices] # N, 3

    # 计算损失
    if loss_type == "point_to_point":
        loss = torch.mean(torch.sum((transformed_ref_pc[valid_mask] - matched_target_pc[valid_mask]) ** 2, dim=1))
    elif loss_type == "point_to_plane":
        diffs = transformed_ref_pc - matched_target_pc
        matched_normals = target_normals[indices]
        proj_errors = torch.sum(diffs[valid_mask] * matched_normals[valid_mask], dim=1)
        loss = torch.mean(proj_errors ** 2)

    return loss


if __name__ == "__main__":
    def test_differentiable_icp_loss():
        # 模拟参考点云和目标点云
        ref_pc = np.random.rand(6000, 3).astype(np.float32)  # 参考点云 (600, 3)
        tgt_pc = np.random.rand(8000, 3).astype(np.float32)  # 目标点云 (800, 3)

        # 模拟 joint_axis_scaled 作为优化参数
        joint_axis_scaled = torch.tensor([1, 0.5, 1], requires_grad=True, device='cuda')

        # 计算损失
        loss = icp_loss_torch(
            joint_axis_scaled, ref_pc, tgt_pc,
            target_normals=None,
            loss_type="point_to_plane", 
            joint_type="prismatic", 
            coor_valid_distance=10
        )

        # 打印损失值
        print(f"ICP Loss: {loss.item()}")

    # 运行测试
    test_differentiable_icp_loss()