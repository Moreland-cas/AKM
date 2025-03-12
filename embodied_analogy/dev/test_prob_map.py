import numpy as np

def generate_cosine_similarity_map(H, W):
    # 随机生成余弦相似度值在 -1 到 1 之间
    return np.random.uniform(-1, 1, (H, W))

def normalize_using_linear(cosine_similarity_map):
    # 使用 (cos + 1) / 2 归一化
    normalized_map = (cosine_similarity_map + 1) / 2
    # 转换为概率图
    probability_map = normalized_map / np.sum(normalized_map)
    return probability_map

def normalize_cos_map_exp(cosine_similarity_map, sharpen_factor=5):
    """
    对于输入的 cosine_similarity_map 进行归一化变为一个概率分布
    sharpen_factor: 用于调整分布的相对大小, sharpen_factor 越大分布越尖锐
    """
    # 使用 exp(cos) 计算
    exp_map = np.exp(cosine_similarity_map * sharpen_factor)
    # 转换为概率图
    probability_map = exp_map / np.sum(exp_map)
    return probability_map

def main(H, W):
    # 生成余弦相似度图
    cosine_similarity_map = generate_cosine_similarity_map(H, W)
    print(cosine_similarity_map)
    # 使用线性归一化方法
    prob_map_linear = normalize_using_linear(cosine_similarity_map)
    print("使用线性归一化方法的概率图:")
    print(prob_map_linear)
    
    # 使用指数归一化方法
    prob_map_exponential = normalize_using_exponential(cosine_similarity_map)
    print("\n使用指数归一化方法的概率图:")
    print(prob_map_exponential)

# 示例输入
H = 3  # 高度
W = 3  # 宽度
main(H, W)
