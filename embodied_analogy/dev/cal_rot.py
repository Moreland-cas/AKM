import numpy as np

def rotation_matrix_between_vectors(a, b):
    # 假设满足 R @ a = b, 求R
    
    # 确保 a 和 b 是单位向量
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # 计算旋转轴 u 和旋转角度 theta
    u = np.cross(a, b)  # 旋转轴是 a 和 b 的叉积
    sin_theta = np.linalg.norm(u)
    cos_theta = np.dot(a, b)  # 夹角余弦
    u = u / sin_theta if sin_theta != 0 else u  # 归一化旋转轴

    # 计算旋转矩阵
    I = np.eye(3)
    u_cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])  # 叉积矩阵

    R = I + np.sin(np.arccos(cos_theta)) * u_cross + (1 - cos_theta) * np.dot(u_cross, u_cross)
    return R

# 测试
a = np.array([0, 0, 1])  # 示例单位向量 a
b = np.array([0, 0, -1]) / 5  # 示例单位向量 b
R = rotation_matrix_from_vector(a, b)
print("旋转矩阵 R:\n", R)
