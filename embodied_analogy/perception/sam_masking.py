import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/sam2")

def run_sam_whole(
    rgb_img, 
    positive_points, # np.array([N, 2])
    negative_points,
    visualize=False
):    
    # load sam2 image predictor model
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"

    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt", 
        device=torch.device("cuda")
    )
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb_img)
    
    # 将 point prompt 整理为 SAM2 的输入格式
    input_point = np.concatenate((positive_points, negative_points), axis=0)
    input_label = np.array([1] * len(positive_points) + [0] * len(negative_points))
    
    # 进行预测
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # 按照分数排序
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind] # N, H, W
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    if visualize:
        import napari
        viewer = napari.view_image(rgb_img, rgb=True)
        viewer.add_points(positive_points[:, [1, 0]], face_color="green")
        viewer.add_points(negative_points[:, [1, 0]], face_color="red")
        for i in range(len(masks)):
            viewer.add_labels(masks[i].astype(np.int32), name=f'score={scores[i]}')
        napari.run()
    return masks[0]


if __name__ == "__main__":
    pass