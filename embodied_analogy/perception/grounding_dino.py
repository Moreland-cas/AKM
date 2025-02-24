import os
import numpy as np
# from PyQt5.QtCore import QCoreApplication, QLibraryInfo

# # 在代码开头添加以下配置
# os.environ["QT_DEBUG_PLUGINS"] = "1"  # 开启插件调试
# os.environ["QT_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
import cv2
import torch
from PIL import Image
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T

# import napari
# viewer = napari.Viewer()
# napari.run()

from groundingdino.util.inference import load_model, predict, annotate

# import napari
# viewer = napari.Viewer()
# napari.run()

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
        
    h, w, _ = image_np.shape
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
        annotated_frame_BGR = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame_RGB = cv2.cvtColor(annotated_frame_BGR, cv2.COLOR_BGR2RGB)
        # Image.fromarray(annotated_frame_RGB).show()
        import napari
        viewer = napari.view_image(annotated_frame_RGB, rgb=True)
        viewer.title = "groundingDINO"
        napari.run()
        
    boxes = boxes * torch.Tensor([w, h, w, h])
    bbox_scaled = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy() # N, 4
    bbox_score = logits.numpy()
    
    # Sort bounding boxes by score in descending order
    sorted_indices = np.argsort(bbox_score)[::-1]  # Sort indices by score (descending order)
    bbox_scaled_sorted = bbox_scaled[sorted_indices]
    bbox_score_sorted = bbox_score[sorted_indices]
    
    return bbox_scaled_sorted, bbox_score_sorted



if __name__ == "__main__":
    # test()
    # load_image("/home/zby/Programs/Embodied_Analogy/third_party/GroundingDINO/sapien_test.png")
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(np.random.random((100, 100)))
    # napari.run()
    bbox_scaled, bbox_score = run_groundingDINO(
        image="/home/zby/Programs/Embodied_Analogy/assets/sapien_test.png",
        text_prompt="object",
        visualize=True
    )
    print(bbox_scaled, bbox_score)
