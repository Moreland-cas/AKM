import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from featup.util import norm, unnorm
import torchvision.transforms as T
from featup.plotting import plot_feats, plot_lang_heatmaps
from embodied_analogy.utility.utils import plot_matching, plot_matching_2, nms_selection, match_point_on_featmap
    
def load_featup_dino_model(device="cuda"):
    """
    加载DINO模型和其相应的feature extractor。
    """
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
    return upsampler

def preprocess_image(image_path, resize=448, device="cuda"):
    """
    预处理输入图像,以适配DINO模型的输入要求。
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
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

def match_points_dino_featup(
    image_path_1, 
    image_path_2, 
    uv_1, # [0.3, 0.7]
    # top_k=100, 
    # max_return=5,
    resize=448, 
    device="cuda", 
    # nms_threshold=0.05,
    visualize=False, 
):
    # 分为匹配特征 + nms 两部分
    upsampler = load_featup_dino_model(device)
    
    # 预处理图像并提取特征
    tensor_1 = preprocess_image(image_path_1, resize, device) # shape: (1, feature_dim, h, w)
    tensor_2 = preprocess_image(image_path_2, resize, device)

    hr_feats_1 = extract_features(tensor_1, upsampler)  # shape: (1, feature_dim, h, w)
    hr_feats_2 = extract_features(tensor_2, upsampler)  # shape: (1, feature_dim, h, w)
    
    # to delete
    similarity_map = match_point_on_featmap(hr_feats_1, hr_feats_2, uv_1, visualize) # H, W
    
    # # 采用概率最大的k 个点
    # topk_values, topk_indices = torch.topk(similarity, top_k, largest=True, sorted=False)
    
    # target_points_uv = np.array([[i % w2, i // w2] for i in topk_indices.cpu().numpy()])
    # target_points_uv = target_points_uv / np.array([w2, h2])
    # target_points_probs = similarity[topk_indices].cpu().numpy()
    
    # # 应用NMS选择最终的匹配点
    # final_points_uv, final_probs = nms_selection(target_points_uv, target_points_probs, threshold=nms_threshold, max_points=max_return)
    
    # if visualize:
    #     plot_matching(unnorm(tensor_1)[0], unnorm(tensor_2)[0], hr_feats_1[0], hr_feats_2[0], similarity_map)
    
    # return final_points_uv, final_probs, similarity_map
    return similarity_map

if __name__ == "__main__":
    path1 = "cat.png"
    path2 = "tom.jpg"
    left_point = (0.72, 0.30)  # 左图中归一化到[0, 1]的坐标
    # 执行匹配
    tensor_1, tensor_2, hr_feats_1, hr_feats_2, similarity_map = match_points_dino_featup(path1, path2, left_point)
    plot_matching(unnorm(tensor_1)[0], unnorm(tensor_2)[0], hr_feats_1[0], hr_feats_2[0], similarity_map)
