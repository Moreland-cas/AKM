import torch
from transformers import DINOFeatureExtractor, DINOModel
from PIL import Image
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


# 加载预训练的DINO模型
def load_dino_model(model_name="facebook/dino-vits16"):
    """
    加载DINO模型和其相应的feature extractor。
    """
    feature_extractor = DINOFeatureExtractor.from_pretrained(model_name)
    model = DINOModel.from_pretrained(model_name)
    return feature_extractor, model

def preprocess_image(image_path, feature_extractor):
    """
    预处理输入图像，以适配DINO模型的输入要求。
    """
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def extract_features(image_tensor, model):
    """
    使用DINO模型从图像tensor中提取特征。
    """
    with torch.no_grad():
        outputs = model(**image_tensor)
    # 获取特征向量
    features = outputs.last_hidden_state
    return features

def compute_similarity(features_left, features_right):
    """
    计算两张图像之间的相似度。
    使用余弦相似度，返回匹配概率。
    """
    # 将特征从 batch x seq_length x feature_dim 转换为 seq_length x feature_dim
    left_features = features_left.squeeze(0).cpu().numpy()  # (seq_length, feature_dim)
    right_features = features_right.squeeze(0).cpu().numpy()  # (seq_length, feature_dim)
    
    # 计算两者之间的余弦相似度
    similarity_matrix = cosine_similarity(left_features, right_features)
    return similarity_matrix

def match_points(left_image, right_image, left_point, feature_extractor, model, top_k=5):
    """
    在右图上找到与左图指定点匹配的点。根据相似度分布采样若干个点。
    """
    # 预处理图像并提取特征
    left_input = preprocess_image(left_image, feature_extractor)
    right_input = preprocess_image(right_image, feature_extractor)

    left_features = extract_features(left_input, model)
    right_features = extract_features(right_input, model)

    # 计算两张图像之间的相似度
    similarity_matrix = compute_similarity(left_features, right_features)
    
    # 获取左图指定点的相似度
    u, v = left_point
    left_idx = int(u * left_features.size(1))  # 将归一化坐标转换为特征索引
    
    # 获取该点与右图所有点的匹配概率
    matching_probabilities = similarity_matrix[left_idx]
    
    # 根据相似度排序，选择top_k个匹配点
    sorted_indices = np.argsort(matching_probabilities)[::-1][:top_k]
    sampled_points = [(i // right_features.size(1), i % right_features.size(1)) for i in sorted_indices]

    return sampled_points, matching_probabilities[sorted_indices]

def main():
    # 加载DINO模型
    feature_extractor, model = load_dino_model()

    # 设置左图和右图路径以及左图中的目标点
    left_image_path = "left_image.jpg"
    right_image_path = "right_image.jpg"
    left_point = (0.5, 0.5)  # 左图中归一化到[0, 1]的坐标

    # 执行匹配
    sampled_points, matching_probs = match_points(left_image_path, right_image_path, left_point, feature_extractor, model)

    # 输出匹配的点和对应的匹配概率
    print("Sampled Points:", sampled_points)
    print("Matching Probabilities:", matching_probs)

if __name__ == "__main__":
    main()

