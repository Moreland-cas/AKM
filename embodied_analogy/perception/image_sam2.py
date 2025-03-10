import torch
import numpy as np
from PIL import Image
import random
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage

import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def select_farthest_point(unsatisfied_points, used_same_class_points, used_diff_class_points):
    if not used_same_class_points and not used_diff_class_points:
        return unsatisfied_points[np.random.choice(len(unsatisfied_points))]

    unsatisfied_points_array = np.array(unsatisfied_points)

    # 计算离同类点的平均距离
    if used_same_class_points:
        same_class_array = np.array(list(used_same_class_points))
        same_class_dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - same_class_array[None, :, :], axis=2)
        mean_same_class_dists = np.min(same_class_dists, axis=1)
    else:
        mean_same_class_dists = np.zeros(len(unsatisfied_points))

    # 计算离异类点的平均距离
    if used_diff_class_points:
        diff_class_array = np.array(list(used_diff_class_points))
        diff_class_dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - diff_class_array[None, :, :], axis=2)
        mean_diff_class_dists = np.min(diff_class_dists, axis=1)
    else:
        mean_diff_class_dists = np.zeros(len(unsatisfied_points))
    
    # 综合考虑：平均离同类距离越大越好，离异类距离越远越好
    combined_score = mean_same_class_dists - mean_diff_class_dists
    # combined_score =  - mean_diff_class_dists
    
    farthest_idx = np.argmax(combined_score)
    return unsatisfied_points[farthest_idx]


def select_dense_point(unsatisfied_points):
    if len(unsatisfied_points) == 1:
        return unsatisfied_points[0]

    unsatisfied_points_array = np.array(unsatisfied_points)
    
    # 计算所有点对之间的距离矩阵
    dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - unsatisfied_points_array[None, :, :], axis=2)
    
    # 计算每个点到其他所有点的平均距离，距离越小，密度越大
    mean_dists = np.mean(dists, axis=1)
    
    # 选择平均距离最小的点，即密度最大的点
    dense_idx = np.argmin(mean_dists)
    return unsatisfied_points[dense_idx]


def select_cluster_center_point(points, return_k_points=1):
    if len(points) <= return_k_points:
        return np.arange(len(points))

    if return_k_points == 1:
        k_clusters = 3
    else:
        k_clusters = return_k_points
        
    kmeans = KMeans(n_clusters=min(k_clusters, len(points)), init='k-means++', random_state=0).fit(points)
    centers = kmeans.cluster_centers_

    # 找到最接近每个中心点的样本
    closest_indices = []
    for center in centers:
        closest_idx = np.argmin(np.linalg.norm(points - center, axis=1))
        closest_indices.append(closest_idx)
    
    if return_k_points == 1:
        return np.array([random.choice(closest_indices)])
    else:
        return np.array(closest_indices)

def load_sam2_image_model():
     # 加载 SAM2 模型
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt",
        device=torch.device("cuda")
    )
    return sam2_model
    
# 运行 SAM2 模型，并逐步优化输入点，以提高分割效果
def sam2_on_image(
    rgb_img, # numpy
    positive_points=None,  # np.array([N, 2])
    positive_bbox=None, # np.array([4]), [u_left, v_left, u_right, v_right]
    negative_points=None,
    num_iterations=5,
    acceptable_thr=0.9,
    sam2_image_model=None,
    post_process=False,
    visualize=False
):
    sam2_model = sam2_image_model
    if sam2_model is None:
        sam2_model = load_sam2_image_model()
        
    assert num_iterations >= 1
    # positive points 或者 positive box 至少有一个不为 None
    assert (positive_bbox is not None) or (positive_points is not None)
    
    # 默认处理 None 的情况
    if positive_points is None:
        positive_points = np.empty((0, 2))  # 为空时设置为形状为 (0, 2) 的空数组
    if negative_points is None:
        negative_points = np.empty((0, 2))  # 为空时设置为形状为 (0, 2) 的空数组
        
    if isinstance(positive_points, torch.Tensor):
        positive_points = positive_points.cpu().numpy()
    if isinstance(negative_points, torch.Tensor):
        negative_points = negative_points.cpu().numpy()

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb_img)

    # 选择正样本点的聚类中心和一个随机负样本点作为初始输入
    # initial_neg_idx = -5 # panda_hand
    # if positive_points.shape[0] > 0:
    #     initial_pos_idx = select_cluster_center_point(positive_points, return_k_points=1)
    # else:
    #     initial_pos_idx = []

    # if negative_points.shape[0] > 0:
    #     initial_neg_idx = select_cluster_center_point(negative_points, return_k_points=1)
    # else:
    #     initial_neg_idx = []
        
    initial_pos_idx, initial_neg_idx = [], []
    
    # input_point = np.vstack([positive_points[initial_pos_idx], negative_points[initial_neg_idx]])
    # input_label = np.append(np.ones(len(initial_pos_idx)), np.zeros(len(initial_neg_idx)))
    input_point = np.empty((0, 2))
    input_label = np.empty(0)

    # 记录已使用的点
    # used_positive_points = set(map(tuple, positive_points[initial_pos_idx])) if len(initial_pos_idx) > 0 else set()
    # used_negative_points = set(map(tuple, negative_points[initial_neg_idx])) if len(initial_neg_idx) > 0 else set()
    used_positive_points = set()
    used_negative_points = set()
    
    acceptable_score = acceptable_thr * len(positive_points) if positive_points is not None else 0
        
    cur_best_score = -1e6
    cur_best_mask = None
    tmp_best_mask = None
    last_logits = None

    # 可视化
    if visualize:
        import napari
        viewer = napari.view_image(rgb_img, rgb=True)
        viewer.title = "sam results"
        viewer.add_points(positive_points[:, [1, 0]], face_color="green", name="input positive points")
        viewer.add_points(negative_points[:, [1, 0]], face_color="red", name="input negative points")
            
    for i in range(num_iterations):
        if cur_best_score >= acceptable_score:
            break
        # 进行预测
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=positive_bbox,
            mask_input=last_logits[None, :, :] if last_logits is not None else None,
            multimask_output=True,
        )

        # 根据分数排序
        tmp_best_mask = masks[np.argmax(scores)]
        last_logits = logits[np.argmax(scores)]

        # 分数函数，并返回不满足条件的点
        def score_mask(mask):
            if positive_points.shape[0] == 0 and negative_points.shape[0] == 0:
                return 0, [], []
            positive_int = positive_points.astype(np.int32)
            negative_int = negative_points.astype(np.int32)
            pos_score = np.sum(mask[positive_int[:, 1], positive_int[:, 0]])
            neg_score = np.sum(mask[negative_int[:, 1], negative_int[:, 0]])

            # 找到不满足的点
            unsatisfied_positive = [tuple(pt) for pt in positive_points if mask[int(pt[1]), int(pt[0])] == 0 and tuple(pt) not in used_positive_points]
            unsatisfied_negative = [tuple(pt) for pt in negative_points if mask[int(pt[1]), int(pt[0])] == 1 and tuple(pt) not in used_negative_points]
            # 把 negative_point 的惩罚加重一点, 因为 negative point 本身就少
            return pos_score - 10 * neg_score, unsatisfied_positive, unsatisfied_negative

        # 更新最佳分割结果
        current_score, unsatisfied_positive, unsatisfied_negative = score_mask(tmp_best_mask)
        if current_score > cur_best_score:
            cur_best_mask = tmp_best_mask
            cur_best_score = current_score

        # 优先选择从不满足的点中选择新点进行扩充
        if unsatisfied_negative:
            new_negative_idx = select_cluster_center_point(unsatisfied_negative, return_k_points=1)[0]
            new_negative_point = unsatisfied_negative[new_negative_idx]
            input_point = np.vstack([input_point, new_negative_point])
            input_label = np.append(input_label, 0)
            used_negative_points.add(tuple(new_negative_point))

        # 从不满足的正样本点中选择最远的点
        if unsatisfied_positive:
            new_positive_idx = select_cluster_center_point(unsatisfied_positive, return_k_points=1)[0]
            new_positive_point = unsatisfied_positive[new_positive_idx]
            input_point = np.vstack([input_point, new_positive_point])
            input_label = np.append(input_label, 1)
            used_positive_points.add(tuple(new_positive_point))
            
        if visualize:
            pass
            # viewer.add_labels(cur_best_mask.astype(np.int32), name=f'cur best mask {i}')
            # viewer.add_labels(tmp_best_mask.astype(np.int32), name=f'tmp best mask {i}')
            
            # used_positive_vis = np.array(list(used_positive_points))
            # used_negative_vis = np.array(list(used_negative_points))
    
            # viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
            # viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")
        
    used_positive_vis = np.array(list(used_positive_points))
    used_negative_vis = np.array(list(used_negative_points))
    
    cur_best_mask = cur_best_mask.astype(np.bool_)
    
    # 在这里对于输出的 mask 进行一个 post-processing
    if post_process:
        cur_best_mask = morphology.binary_closing(cur_best_mask, morphology.disk(10))
        cur_best_mask = ndimage.binary_fill_holes(cur_best_mask)   
        
    if visualize:
        viewer.add_labels(cur_best_mask.astype(np.int32), name='cur best mask')
        
        if len(used_positive_vis) > 0:
            viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
        if len(used_negative_vis) > 0:
            viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")
        if positive_bbox is not None:
            u_leftup, v_leftup, u_rightdown, v_rightdown = positive_bbox
            bbox_rect = np.array([
                [u_leftup, v_leftup],
                [u_rightdown, v_leftup],
                [u_rightdown, v_rightdown],
                [u_leftup, v_rightdown],
            ])
            # 在这里 transpose一下, 因为 napari 的坐标是 (v, u)
            bbox_rect = bbox_rect[:, [1, 0]]
            viewer.add_shapes(
                bbox_rect[None], # should be of shape 1, 4, 2
                face_color="transparent",
                edge_color="green",
                edge_width=5,
                name="positive bbox prompt"
            )
        napari.run()

    return cur_best_mask, used_positive_vis, used_negative_vis


if __name__ == "__main__":
    image_pil = Image.open("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/sapien_test.png")
    # image_pil.show() # 1100 x 1100
    image_np = np.array(image_pil.convert("RGB"))
    input_bbox = np.array([0, 0, 1000, 400])
    
    # positive_points = np.array([[550, 550], [551, 554]])
    positive_points = None
    negative_points = np.array([[5, 5]])
    negative_points = None
    sam2_on_image(
        rgb_img=image_np, 
        positive_points=positive_points, 
        positive_bbox=input_bbox,
        negative_points=negative_points, 
        num_iterations=3,
        acceptable_thr=0.9,
        sam2_image_model=load_sam2_image_model(),
        visualize=True
    )