import numpy as np
from embodied_analogy.estimation.utils import *

def point_to_plane_icp_cpu(ref_pc, target_pc, init_transform, mode="prismatic", max_iterations=20, tolerance=1e-6):
    """
    点到平面 ICP 算法，计算刚性变换
    :param ref_pc: numpy 数组，形状 (M, 3)，参考点云
    :param target_pc: numpy 数组，形状 (N, 3)，目标点云
    :param init_transform: 初始变换矩阵，形状 (4, 4)
    :param mode: "revolute" 仅估计旋转, "prismatic" 仅估计平移
    :param max_iterations: 最大迭代次数
    :param tolerance: 终止条件
    :return: 计算出的刚性变换矩阵 (4, 4)
    """
    # 应用初始变换
    ref_pc = (init_transform[:3, :3] @ ref_pc.T + init_transform[:3, 3:]).T
    
    # 计算目标点云的法向量
    normals = compute_normals(target_pc)

    prev_error = np.inf
    transform = np.eye(4)
    
    for i in range(max_iterations):
        # 最近邻匹配
        indices = find_correspondences(ref_pc, target_pc)
        matched_target = target_pc[indices]
        matched_normals = normals[indices]

        # 计算点到平面误差
        diffs = ref_pc - matched_target
        errors = np.sum(diffs * matched_normals, axis=1)
        
        # 组装最小二乘矩阵
        A = []
        b = []
        
        for j in range(len(ref_pc)):
            n = matched_normals[j]
            p = ref_pc[j]
            
            if mode == "revolute":
                # 旋转模式：约束方程 n_i^T * (R * p_i - q_i) = 0
                A.append(np.cross(p, n))  # 叉乘用于旋转约束
            elif mode == "prismatic":
                # 平移模式：约束方程 n_i^T * (p_i + t - q_i) = 0
                A.append(n)
            
            b.append(-errors[j])
        
        A = np.vstack(A)
        b = np.array(b)

        # 求解最小二乘 Ax = b
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        if mode == "revolute":
            # 计算旋转矩阵
            theta = np.linalg.norm(x)
            if theta > 0:
                axis = x / theta
                skew_sym = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R_update = np.eye(3) + np.sin(theta) * skew_sym + (1 - np.cos(theta)) * (skew_sym @ skew_sym)
            else:
                R_update = np.eye(3)
            
            update_transform = np.eye(4)
            update_transform[:3, :3] = R_update
        elif mode == "prismatic":
            # 计算平移向量
            t_update = x
            update_transform = np.eye(4)
            update_transform[:3, 3] = t_update
        
        # 应用变换
        transform = update_transform @ transform
        ref_pc = (update_transform[:3, :3] @ ref_pc.T + update_transform[:3, 3:]).T

        # 计算误差
        mean_error = np.mean(errors**2)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return transform @ init_transform

# ===== 测试代码 =====
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    target_pc = np.random.rand(1000, 3) * 10  # 目标点云

    # 生成已知变换（仅旋转或仅平移）
    true_transform = np.eye(4)
    mode = "rotation"  # 选择 "rotation" 或 "translation"
    
    if mode == "rotation":
        theta = np.pi / 6  # 30 度旋转
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        true_transform[:3, :3] = R_true
    elif mode == "translation":
        true_transform[:3, 3] = [2, -1, 3]  # 纯平移
    
    ref_pc = (true_transform[:3, :3] @ target_pc.T + true_transform[:3, 3:]).T  # 变换后的点云

    # 设定初始估计
    init_transform = np.eye(4)
    
    # 运行点到平面 ICP
    estimated_transform = point_to_plane_icp(ref_pc, target_pc, init_transform, mode=mode)

    print("真实变换矩阵:\n", true_transform)
    print("估计变换矩阵:\n", estimated_transform)
