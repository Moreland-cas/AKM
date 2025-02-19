from scipy.spatial import cKDTree
import numpy as np
import time

def find_correspondences(ref_pc, target_pc, max_distance=0.01, leafsize=30, n_jobs=1):
    """
    使用 KD-Tree 进行最近邻搜索，找到参考点云 ref_pc 在目标点云 target_pc 中的最近邻
    :param ref_pc: numpy 数组，形状 (M, 3)，参考点云
    :param target_pc: numpy 数组，形状 (N, 3)，目标点云
    :param max_distance: 最大距离，超过这个距离的点会被忽略
    :param leafsize: KD-Tree 构建时的 leafsize 参数
    :param n_jobs: 使用的并行线程数，默认为 1 (单线程)，-1 为所有可用核心
    :return: 匹配的索引数组和有效掩码
    """
    # 创建 KD-Tree 对象，调整 leafsize 和并行查询线程数
    tree = cKDTree(target_pc, leafsize=leafsize)
    
    # 查询最近邻，k=1 表示只返回最近的一个邻居
    distances, indices = tree.query(ref_pc, k=1, workers=n_jobs)
    
    # 筛选出满足最大距离条件的匹配
    valid_mask = distances < max_distance
    
    return indices, valid_mask

# 示例用法
ref_pc = np.random.random((10000, 3))  # 参考点云（10000个点）
target_pc = np.random.random((10000, 3))  # 目标点云（10000个点）

# 不同的 leafsize 和 n_jobs 配置
leafsizes = [10, 30, 50, 100]  # 叶子节点的大小
n_jobs_list = [1, 2, 4, 8, 16, -1]  # 单线程与多线程

# 记录每种配置的查询时间
results = []

# 循环遍历不同的 leafsize 和 n_jobs 设置，测试查询时间
for leafsize in leafsizes:
    for n_jobs_value in n_jobs_list:
        total_time = 0.0
        
        # 执行 100 次查询，计算平均时间
        for _ in range(1000):
            start_time = time.time()
            
            # 调用 find_correspondences 函数进行查询
            indices, valid_mask = find_correspondences(ref_pc, target_pc, max_distance=0.1, leafsize=leafsize, n_jobs=n_jobs_value)
            
            end_time = time.time()
            
            # 累加查询时间
            total_time += (end_time - start_time)
        
        # 计算平均查询时间
        avg_time = total_time
        
        # 记录结果
        results.append({
            'leafsize': leafsize,
            'n_jobs': n_jobs_value,
            'avg_time': avg_time
        })

# 打印结果表格
print("测试结果（查询时间，单位：秒）")
header = ["n_jobs"] + [str(n) for n in n_jobs_list]
table = {n: [] for n in leafsizes}

# 填充表格数据
for leafsize in leafsizes:
    for n_jobs_value in n_jobs_list:
        avg_time = next(result['avg_time'] for result in results if result['leafsize'] == leafsize and result['n_jobs'] == n_jobs_value)
        table[leafsize].append(f"{avg_time:.2f}s")

# 打印表头
print(" ".join(header))

# 打印每一行
for leafsize in leafsizes:
    row = [str(leafsize)] + table[leafsize]
    print(" ".join(row))
