import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/sam2")


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

# 函数功能：培训并运行 SAM2 模型，并逐步优化输入点，以提高分割效果
def run_sam_whole(
    rgb_img,
    positive_points,  # np.array([N, 2])
    negative_points,
    num_iterations=3,
    visualize=False
):
    assert num_iterations >= 1
    if isinstance(positive_points, torch.Tensor):
        positive_points = positive_points.cpu().numpy()
    if isinstance(negative_points, torch.Tensor):
        negative_points = negative_points.cpu().numpy()

    # 加载 SAM2 模型
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt",
        device=torch.device("cuda")
    )
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb_img)

    # 随机抽取一个正样本点和一个负样本点作为初始输入
    initial_pos_idx = np.random.choice(len(positive_points))
    initial_neg_idx = np.random.choice(len(negative_points))
    
    input_point = np.vstack([positive_points[initial_pos_idx], negative_points[initial_neg_idx]])
    input_label = np.array([1, 0])

    # 记录已使用的点
    used_positive_points = set([tuple(positive_points[initial_pos_idx])])
    used_negative_points = set([tuple(negative_points[initial_neg_idx])])
    
    cur_best_score = -1e6
    cur_best_mask = None
    tmp_best_mask = None
    last_logits = None

    # 可视化
    if visualize:
        import napari
        viewer = napari.view_image(rgb_img, rgb=True)
        # viewer.add_points(positive_points[:, [1, 0]], face_color="green", name="input positive points")
        # viewer.add_points(negative_points[:, [1, 0]], face_color="red", name="input negative points")
            
    for i in range(num_iterations):
        # 进行预测
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            # mask_input=tmp_best_mask[None, :, :] if tmp_best_mask is not None else None, # TODO 这里需要 ablate 一下到底是 cur_best 好还是 tmp_best 好, 还是不用好
            mask_input=last_logits[None, :, :] if last_logits is not None else None,
            multimask_output=True,
        )

        # 根据分数排序
        # sorted_ind = np.argsort(scores)[::-1]
        # masks = masks[sorted_ind]
        # scores = scores[sorted_ind]
        # logits = logits[sorted_ind]

        tmp_best_mask = masks[np.argmax(scores)]
        last_logits = logits[np.argmax(scores)]

        # 分数函数，并返回不满足条件的点
        def score_mask(mask):
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

        # 增加点, 筛选点的逻辑到底是什么？？
        # 增加点：找不满足的点
        # 应该找不满足的点中密度最高的那个（最处于中心的那个）
        
        # 优先选择从不满足的点中选择新点进行扩充
        if unsatisfied_negative:
            # new_negative_point = np.array(select_farthest_point(unsatisfied_negative, used_negative_points, used_positive_points))
            new_negative_point = np.array(select_dense_point(unsatisfied_negative))
            input_point = np.vstack([input_point, new_negative_point])
            input_label = np.append(input_label, 0)
            used_negative_points.add(tuple(new_negative_point))

        # 从不满足的正样本点中选择最远的点
        if unsatisfied_positive:
            # new_positive_point = np.array(select_farthest_point(unsatisfied_positive, used_positive_points, used_negative_points))
            new_positive_point = np.array(select_dense_point(unsatisfied_positive))
            input_point = np.vstack([input_point, new_positive_point])
            input_label = np.append(input_label, 1)
            used_positive_points.add(tuple(new_positive_point))
            
        if visualize:
            viewer.add_labels(cur_best_mask.astype(np.int32), name=f'cur best mask {i}')
            # viewer.add_labels(tmp_best_mask.astype(np.int32), name=f'tmp best mask {i}')
            
            # used_positive_vis = np.array(list(used_positive_points))
            # used_negative_vis = np.array(list(used_negative_points))
    
            # viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
            # viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")

    if visualize:
        used_positive_vis = np.array(list(used_positive_points))
        used_negative_vis = np.array(list(used_negative_points))

        viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
        viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")
        napari.run()

    return cur_best_mask




if __name__ == "__main__":
    pass