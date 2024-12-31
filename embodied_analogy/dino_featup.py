import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union
from featup.util import norm, unnorm
import torchvision.transforms as T
import torch.nn.functional as F
from featup.plotting import plot_feats, plot_lang_heatmaps
from embodied_analogy.utils import plot_matching, nms_selection
    
def load_featup_dino_model(device="cuda"):
    """
    加载DINO模型和其相应的feature extractor。
    """
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
    return upsampler

def preprocess_image(image_path, resize=224, device="cuda", is_pil=False):
    """
    预处理输入图像,以适配DINO模型的输入要求。
    """
    if not is_pil:
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")
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

def match_point_with_featmap(
    feat_1: torch.Tensor, 
    feat_2: torch.Tensor,
    uv_1: List[float],
    visualize: bool=False, 
):
    """_summary_

    Args:
        feat_1 (torch.Tensor): 1, C, H, W
        feat_2 (torch.Tensor): 1, C, H, W
        uv_1 (List[float]): [u, v], u and v are in [0, 1]
        visualize (bool): whether to visualize the matching probability map
    return:
        prob_map (torch.Tensor): H, W, range [0, 1]
    """
    pass

def match_points_dino_featup(
    image_path_1, 
    image_path_2, 
    left_point, 
    top_k=100, 
    max_return=5,
    resize=224, 
    device="cuda", 
    is_pil=False, 
    nms_threshold=0.05,
    visualize=False, 
    ):
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
    # 分为匹配特征 + nms 两部分
    upsampler = load_featup_dino_model(device)
    
    # 预处理图像并提取特征
    tensor_1 = preprocess_image(image_path_1, resize, device, is_pil) # shape: (1, feature_dim, h, w)
    tensor_2 = preprocess_image(image_path_2, resize, device, is_pil)

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
    
    # 采用概率最大的k 个点
    topk_values, topk_indices = torch.topk(normalized_probs, top_k, largest=True, sorted=False)
    
    target_points_uv = np.array([[i % w2, i // w2] for i in topk_indices.cpu().numpy()])
    target_points_uv = target_points_uv / np.array([w2, h2])
    target_points_probs = similarity[topk_indices].cpu().numpy()
    
    # 应用NMS选择最终的匹配点
    final_points_uv, final_probs = nms_selection(target_points_uv, target_points_probs, threshold=nms_threshold, max_points=max_return)
    
    if visualize:
        plot_matching(unnorm(tensor_1)[0], unnorm(tensor_2)[0], hr_feats_1[0], hr_feats_2[0], similarity_map)
    
    return final_points_uv, final_probs, similarity_map

if __name__ == "__main__":
    path1 = "cat1.jpg"
    path2 = "tom.jpg"
    left_point = (0.6, 0.2)  # 左图中归一化到[0, 1]的坐标
    # 执行匹配
    tensor_1, tensor_2, hr_feats_1, hr_feats_2, similarity_map = match_points_dino_featup(path1, path2, left_point)

    plot_matching(unnorm(tensor_1)[0], unnorm(tensor_2)[0], hr_feats_1[0], hr_feats_2[0], similarity_map)
    