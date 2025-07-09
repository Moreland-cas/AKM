import torch
from embodied_analogy.utility.perception.image_sam2 import sam2_on_image, load_sam2_image_model
from embodied_analogy.utility.perception.grounding_dino import run_groundingDINO, load_groundingDINO_model

@torch.no_grad()
def run_grounded_sam(
    rgb_image,
    obj_description,
    positive_points=None,  # np.array([N, 2])
    negative_points=None,
    num_iterations=3,
    acceptable_thr=0.9,
    dino_model=None,
    sam2_image_model=None,
    post_process_mask=True,
    visualize=False,
):
    initial_bboxs, initial_bbox_scores = run_groundingDINO(
        image=rgb_image,
        obj_description=obj_description,
        dino_model=dino_model,
        visualize=False,
    )
    # TODO: 这里也许可以有更复杂的 box 处理方式, 比如合并多个 box 以解决物体在操作过程中进行了分离的情况
    initial_bbox = initial_bboxs[0]

    # 然后根据 initial_bbox 得到 initial_mask
    initial_mask, _, _ = sam2_on_image(
        rgb_img=rgb_image, 
        positive_points=positive_points,  
        positive_bbox=initial_bbox, 
        negative_points=negative_points,
        num_iterations=num_iterations,
        acceptable_thr=acceptable_thr,
        sam2_image_model=sam2_image_model,
        post_process=post_process_mask,
        visualize=visualize,
    )
    # torch.cuda.empty_cache()
    return initial_bbox, initial_mask

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    initial_bbox, initial_mask = run_grounded_sam(
        rgb_image=np.asarray(Image.open("/home/zby/Programs/Embodied_Analogy/assets/unit_test/gsam/cat.jpeg").convert("RGB")),
        obj_description="drawer",
        positive_points=None, 
        negative_points=None,
        num_iterations=3,
        acceptable_thr=0.9,
        dino_model=load_groundingDINO_model(),
        sam2_image_model=load_sam2_image_model(),
        post_process_mask=True,
        visualize=False
    )