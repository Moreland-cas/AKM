import torch

from akm.utility.perception.image_sam2 import sam2_on_image, load_sam2_image_model
from akm.utility.perception.grounding_dino import run_groundingDINO, load_groundingDINO_model


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
    initial_bbox = initial_bboxs[0]

    # get initial_mask from  initial_bbox
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
    torch.cuda.empty_cache()
    return initial_bbox, initial_mask


if __name__ == "__main__":
    import cv2
    img = cv2.imread("/home/zhangboyuan/Programs/AKM/assets/dev/cat.jpg")
    import time
    for i in range(10):
        start_time = time.time()
        # run_groundingDINO(
        #     image=img,
        #     obj_description="cat",
        #     dino_model=None,
        #     visualize=False,
        # )
        run_grounded_sam(
            rgb_image=img,
            obj_description="cat",
            positive_points=None,  # np.array([N, 2])
            negative_points=None,
            num_iterations=3,
            acceptable_thr=0.9,
            dino_model=None,
            sam2_image_model=None,
            post_process_mask=True,
            visualize=False,
        )
        end_time = time.time()
        print(end_time - start_time)
        time.sleep(0.5)
