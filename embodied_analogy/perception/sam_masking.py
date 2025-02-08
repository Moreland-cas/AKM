import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/sam2")

# 函数功能：培训并运行 SAM2 模型，并逐步优化输入点，以提高分割效果
def run_sam_whole(
    rgb_img,
    positive_points,  # np.array([N, 2])
    negative_points,
    num_iterations=5,
    visualize=False
):
    # TODO： 1) 当前计算 score 的方式有问题, 因为 positive_points 的数目远大于 negative_points
    # 2) 当前的加点逻辑是优先加 positive 的点, 但是不是需要优先加 negative ?
    # 3) 加点时候应该优先加 unsatisfied+point 中, 离 used_points 最远的点
    # 4) 一次也不一定只是加一个点吧？
    
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

    # 随机抽取一个正样本点作为初始输入
    initial_idx = np.random.choice(len(positive_points))
    input_point = positive_points[initial_idx:initial_idx+1]
    input_label = np.array([1])

    # 记录已使用的点
    used_positive_points = set([tuple(input_point[0])])
    used_negative_points = set()

    cur_best_score = -len(negative_points)
    cur_best_mask = None
    tmp_best_mask = None
    last_logits = None

    # 可视化
    if visualize:
        import napari
        viewer = napari.view_image(rgb_img, rgb=True)
        viewer.add_points(positive_points[:, [1, 0]], face_color="green", name="input positive points")
        viewer.add_points(negative_points[:, [1, 0]], face_color="red", name="input negative points")
            
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

            return pos_score - neg_score, unsatisfied_positive, unsatisfied_negative

        # 更新最佳分割结果
        current_score, unsatisfied_positive, unsatisfied_negative = score_mask(tmp_best_mask)
        if current_score > cur_best_score:
            cur_best_mask = tmp_best_mask
            cur_best_score = current_score

        # 优先选择从不满足的点中选择新点进行扩充
        if unsatisfied_positive and (np.random.rand() > 0.5 or not unsatisfied_negative):
            new_point = np.array(unsatisfied_positive[np.random.choice(len(unsatisfied_positive))])
            input_point = np.vstack([input_point, new_point])
            input_label = np.append(input_label, 1)
            used_positive_points.add(tuple(new_point))
        elif unsatisfied_negative:
            new_point = np.array(unsatisfied_negative[np.random.choice(len(unsatisfied_negative))])
            input_point = np.vstack([input_point, new_point])
            input_label = np.append(input_label, 0)
            used_negative_points.add(tuple(new_point))
        
        if visualize:
            viewer.add_labels(cur_best_mask.astype(np.int32), name=f'cur best mask {i}')
            viewer.add_labels(tmp_best_mask.astype(np.int32), name=f'tmp best mask {i}')
            
            used_positive_vis = np.array(list(used_positive_points))
            used_negative_vis = np.array(list(used_negative_points))
    
            viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
            if len(used_negative_vis) > 0:
                viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")

    if visualize:
        napari.run()

    return cur_best_mask




if __name__ == "__main__":
    pass