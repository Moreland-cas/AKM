import os
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate, load_image

def run_groundingDINO(
    image,
    text_prompt,
    visualize=False
):
    groundingDINO_home = "/home/zby/Programs/Embodied_Analogy/third_party/GroundingDINO"
    WEIGHTS_PATH = os.path.join(groundingDINO_home, "weights", "groundingdino_swint_ogc.pth")
    CONFIG_PATH = os.path.join(groundingDINO_home, "groundingdino/config/GroundingDINO_SwinT_OGC.py")

    if isinstance(image, str):
        image_pil = Image.open(image).convert("RGB")
        image_np = np.asarray(image_pil)
    elif isinstance(image, np.ndarray):
        image_np = image
        image_pil = Image.fromarray(image_np)
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGB")
        image_np = np.asarray(image)
    
    # 处理 image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_pil, None) # torch.Tensor

    # load model
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    boxes, logits, phrases = predict(
        model=model, 
        image=image_transformed, 
        caption=text_prompt, 
        box_threshold=0.35, 
        text_threshold=0.25
    )
    if visualize:
        annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
        Image.fromarray(annotated_frame).show()


if __name__ == "__main__":
    # load_image("/home/zby/Programs/Embodied_Analogy/third_party/GroundingDINO/sapien_test.png")
    run_groundingDINO(
        image="/home/zby/Programs/Embodied_Analogy/third_party/GroundingDINO/sapien_test.png",
        text_prompt="drawer",
        visualize=True
    )