import torch
from PIL import Image
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from featup.util import norm, unnorm
import torchvision.transforms as T
import torch.nn.functional as F
from featup.plotting import plot_feats, plot_lang_heatmaps
from pytorch_lightning import seed_everything
from featup.util import pca, remove_axes
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_matching(image1, image2, hr1, hr2, span):
    seed_everything(0)
    [hr_feats_pca_1, hr_feats_pca_2], _ = pca([hr1.unsqueeze(0), hr2.unsqueeze(0)])
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(image1.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image 1")
    ax[1].imshow(image2.permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Image 2")
    ax[2].imshow(hr_feats_pca_1[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Features 1")
    ax[3].imshow(hr_feats_pca_2[0].permute(1, 2, 0).detach().cpu())
    ax[3].set_title("Features 2")
    ax[4].imshow(span.detach().cpu(), cmap='jet') # "viridis"
    ax[4].set_title("Span")
    remove_axes(ax)
    plt.show()
    
def load_featup_dino_model(device="cuda"):
    """
    加载DINO模型和其相应的feature extractor。
    """
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
    return upsampler

def preprocess_image(image_path, resize=224, device="cuda"):
    """
    预处理输入图像,以适配DINO模型的输入要求。
    """
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        # T.Resize(resize), # resize the lower edge to input_size
        T.Resize((resize, resize)),
        T.CenterCrop((resize, resize)),
        T.ToTensor(),
        norm
    ])
    image_tensor = transform(image).unsqueeze(0).to(device) # 1 c h w
    return image_tensor

def extract_features(image_tensor, model):
    """
    使用DINO模型从图像tensor中提取特征。
    """
    with torch.no_grad():
        hr_feats = model(image_tensor) # 1 384 h w
    return hr_feats

def match_points(image_path_1, image_path_2, left_point, top_k=5, resize=224, device="cuda"):
    """
    在右图上找到与左图指定点匹配的点。根据相似度分布采样若干个点。
    
    Args:
    - image_path_1 (str): 左图像路径
    - image_path_2 (str): 右图像路径
    - left_point (tuple): 左图上的目标点坐标 (u, v)，范围 [0, 1]
    - model (nn.Module): 特征提取模型
    - top_k (int): 需要返回的最匹配的点数
    - resize (int): 输入图像的缩放尺寸
    - device (str): 计算设备，默认使用 "cuda"
    
    Returns:
    - sampled_points (list): 在右图中找到的与左图指定点匹配的点坐标
    - matching_probabilities (Tensor): top_k个匹配点的相似度概率
    """
    upsampler = load_featup_dino_model(device)
    
    # 预处理图像并提取特征
    tensor_1 = preprocess_image(image_path_1, resize, device) # shape: (1, feature_dim, h, w)
    tensor_2 = preprocess_image(image_path_2, resize, device)

    hr_feats_1 = extract_features(tensor_1, upsampler)  # shape: (1, feature_dim, h, w)
    hr_feats_2 = extract_features(tensor_2, upsampler)  # shape: (1, feature_dim, h, w)

    # 获取左图指定点的坐标 (u, v)，并转换为像素坐标
    u, v = left_point
    h1, w1 = hr_feats_1.shape[2], hr_feats_1.shape[3]  # 获取高和宽
    left_pixel_x = int(u * w1)  # 对应的像素坐标
    left_pixel_y = int(v * h1)  # 对应的像素坐标
    
    # 获取左图指定点的特征
    left_point_feature = hr_feats_1[0, :, left_pixel_y, left_pixel_x]  # (feature_dim, )

    # 计算左图指定点与右图所有点的余弦相似度
    # 将左图指定点的特征扩展到右图的每个位置进行比较
    right_features = hr_feats_2.view(hr_feats_2.size(1), -1).T  # shape: (seq_length, feature_dim)

    # 计算余弦相似度
    similarity = F.cosine_similarity(left_point_feature.unsqueeze(0), right_features, dim=1) # 65536
    # 根据相似度概率分布，进行softmax归一化
    normalized_probs = F.softmax(similarity, dim=0) # 65536

    # 提取匹配点的坐标
    h2, w2 = hr_feats_2.shape[2], hr_feats_2.shape[3]  # 获取高和宽
    similarity_map = similarity.reshape(h2, w2)
    
    # 采样 top_k 个点
    sampled_indices = torch.multinomial(normalized_probs, top_k, replacement=False)
    sampled_points = [(i // w2, i % w2) for i in sampled_indices.cpu().numpy()]

    # 返回匹配点和对应的相似度概率
    # return sampled_points, similarity[sampled_indices].cpu().numpy()
    return tensor_1, tensor_2, hr_feats_1, hr_feats_2, similarity_map

if __name__ == "__main__":
    path1 = "cat1.jpg"
    path2 = "tom.jpg"
    left_point = (0.6, 0.2)  # 左图中归一化到[0, 1]的坐标
    # 执行匹配
    tensor_1, tensor_2, hr_feats_1, hr_feats_2, similarity_map = match_points(path1, path2, left_point)

    plot_matching(unnorm(tensor_1)[0], unnorm(tensor_2)[0], hr_feats_1[0], hr_feats_2[0], similarity_map)
    
    