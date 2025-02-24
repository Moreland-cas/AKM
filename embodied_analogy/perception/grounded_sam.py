from embodied_analogy.perception.sam_masking import run_sam_whole
from embodied_analogy.perception.grounding_dino import run_groundingDINO

def run_grounded_sam(
    rgb_image,
    text_prompt,
    positive_points=None,  # np.array([N, 2])
    negative_points=None,
    num_iterations=5,
    acceptable_thr=0.9,
    visualize=False,
):
    
    initial_bboxs, initial_bbox_scores = run_groundingDINO(
        image=rgb_image,
        text_prompt=text_prompt,
        visualize=visualize
    )
    initial_bbox = initial_bboxs[0]

    # 然后根据 initial_bbox 得到 initial_mask
    initial_mask, _, _ = run_sam_whole(
        rgb_img=rgb_image, 
        positive_points=positive_points,  
        positive_bbox=initial_bbox, 
        negative_points=negative_points,
        num_iterations=num_iterations,
        acceptable_thr=acceptable_thr,
        visualize=visualize
    )
    return initial_bbox, initial_mask
    