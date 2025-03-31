import os
from subset_retrieval.gpt_core.chatbot import Chatbot
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from vision.clip_encoder import ClipModel
import pickle
import torch
from vision.featurizer.run_featurizer import extract_ft, match_fts, sample_highest
from vision.featurizer.utils.correspondence import get_distance_bbnn, get_distance_imd
from tqdm import tqdm
# from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image, crop_image
from embodied_analogy.utility.utils import draw_points_on_image

MAX_IMD_RANKING_NUM = 30 # change this for different levels of efficiency

def segment_images(frames, trajs, obj_description, visualize=False):
    from embodied_analogy.utility.perception.grounded_sam import run_grounded_sam
    # 重写了这个函数, 用 embodied_analogy 的 grounded sam 来完成
    masked_frames = []
    frame_masks = []
    for idx, frame in enumerate(frames):
        # masks = inference_one_image(frame, grounded_dino_model, sam_predictor, box_threshold=box_threshold, text_threshold=text_threshold, text_prompt=text_prompt, device=device).cpu().numpy()
        _, mask = run_grounded_sam(
            rgb_image=frame,
            obj_description=obj_description, # drawer
            positive_points=None,
            negative_points=None, # 如果有遮挡的话, 在这里需要加入 negative points, 默认在初始时没有遮挡
            num_iterations=5,
            acceptable_thr=0.9,
            visualize=visualize,
        )
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        # if no object is detected or the contact point is not in the mask, return the original image
        if mask.sum() == 0:
            mask = np.ones_like(mask)
            
        if trajs is not None:
            cp = trajs[idx][0]
            if mask[..., 0][int(cp[1]), int(cp[0])] == 0:
                mask = np.ones_like(mask)

        masked_frame = frame * mask + 255 * (1 - mask)
        masked_frames.append(masked_frame)
        frame_masks.append(mask)
        
    return masked_frames, frame_masks

def concat_images_with_lines(img_list):
    # Assume all images are the same size for simplicity
    img_width, img_height = img_list[0].size
    
    # Create a new image with a white background
    total_width = img_width * 3
    total_height = img_height * 2
    new_img = Image.new('RGB', (total_width, total_height), 'white')
    
    # Font for numbering images
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(img_height/10))
    except IOError as ioe:
        print('=> [ERROR] in font:', ioe)
        font = ImageFont.load_default()
    
    # Draw instance to draw lines and text
    draw = ImageDraw.Draw(new_img)
    
    color_set = ['red', 'green', 'blue', 'brown', 'purple', 'orange']

    # Place images and draw red lines and numbers
    for index, img in enumerate(img_list):
        img = img.resize((img_width, img_height), Image.BILINEAR)
        x = index % 3 * img_width
        y = index // 3 * img_height
        new_img.paste(img, (x, y))
        number = str(index)
        if index == 0:
            number = 'src'
        draw.text((x + img_width/20, y + img_height/20), number, font=font, fill=color_set[index])
    
    # Draw red lines
    for i in range(1, 3):
        draw.line([(i * img_width, 0), (i * img_width, total_height)], fill='red', width=5)
    for i in range(1, 2):
        draw.line([(0, i * img_height), (total_width, i * img_height)], fill='red', width=5)
    
    return new_img

def get_bounding_box(binary_array):
    # Find the rows and columns where there are non-zero elements
    rows = np.any(binary_array, axis=1)
    cols = np.any(binary_array, axis=0)
    
    # Find the indices of the first and last non-zero row and column
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def crop_image(img, mask, traj=[], margin=50):
    '''
    crop img based on the seg mask, and transform the src_pos_list to the cropped image space
    @param
    :img: np.array, (h, w, 3)
    :mask: np.array, (h, w, 3) # the last channel is repeated 3 times
    :src_pos_list: list of tuple, [(x1, y1), ...]
    '''
    bbox = get_bounding_box(mask[..., 0])
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img.shape[1], x_max + margin)
    y_max = min(img.shape[0], y_max + margin)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_traj = []
    for p in traj:
        x, y = p
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue
        x = x - x_min
        y = y - y_min
        cropped_traj.append((x, y))
    return cropped_img, cropped_mask, cropped_traj, (x_min, y_min, x_max, y_max)

def crop_images(masked_frames, masks, trajs=None):
    """
        masked_frames: list of (H, W, 3)
        masks: list of (H, W, 3)
        trajs: list of (u, v)
    """
    cropped_frames = []
    cropped_masks = []
    cropped_trajs = []
    cropped_regions = []
    for idx in range(len(masked_frames)):
        masked_frame = masked_frames[idx]
        mask = masks[idx]
        traj = trajs[idx] if trajs is not None else []
        cropped_image, cropped_mask, cropped_traj, cropped_region = crop_image(masked_frame, mask, traj, margin=100)
        cropped_frames.append(cropped_image)
        cropped_masks.append(cropped_mask)
        cropped_trajs.append(cropped_traj)
        cropped_regions.append(cropped_region)
        
    return cropped_frames, cropped_masks, cropped_trajs, cropped_regions

def get_end_start_direction(trajs):
    dirs = []
    for traj in trajs:
        end_start_dir = traj[-1] - traj[0]
        dirs.append(end_start_dir)
    return dirs

class SubsetRetrievePipeline:
    def __init__(self, subset_dir, save_root=None, topk=5, lang_mode='clip', crop=True, data_source=None) -> None:
        self.subset_dir = subset_dir
        self.save_root = save_root
        self.topk = topk
        self.crop = crop

        # self.grounded_dino_model, self.sam_predictor = prepare_gsam_model(device="cuda")
        self.clip_model = ClipModel()

        # 1. construct a task list
        task_list_droid = os.listdir(os.path.join(subset_dir, "droid"))
        task_list_hoi4d = os.listdir(os.path.join(subset_dir, "HOI4D"))
        task_list_customize = os.listdir(os.path.join(subset_dir, "customize"))
        if data_source == "droid":
            task_list = task_list_droid
        elif data_source == "HOI4D":
            task_list = task_list_hoi4d
        elif data_source == "customize":
            task_list = task_list_customize
        else:
            raise ValueError("data_source should be 'droid' or 'HOI4D' or 'customize'")

        self.data_source = data_source

        self.task_list = [task.replace("_", " ") for task in task_list]

        self.lang_mode = lang_mode
        if lang_mode == 'gpt':
            self.gpt_chatbot = Chatbot(api_key=None) # NEED TO FILL IN YOUR API KEY LIKE 'sk-...'
        elif lang_mode == 'clip':
            pass
        else:
            raise ValueError("lang_mode should be 'gpt' or 'clip'")

    def language_retrieve(self, current_task):
        if self.lang_mode == 'gpt':
            retrieved_task = self.gpt_chatbot.task_retrieval(self.task_list, current_task)
        elif self.lang_mode == 'clip':
            retrieved_task = self.clip_lang_retrieve(current_task)

        return retrieved_task
    
    def clip_lang_retrieve(self, current_task):
        task_embeddings = []
        for task in self.task_list:
            task_embeddings.append(self.clip_model.get_text_feature(task))

        current_task_embedding = self.clip_model.get_text_feature(current_task)

        similarity = self.clip_model.compute_similarity(current_task_embedding, torch.cat(task_embeddings, dim=0))

        max_index = similarity.argmax()

        return self.task_list[max_index]

    def segment_objects(self, imgs, obj_name, trajs=None):
        masked_frames, masks = segment_images(
            frames=imgs,
            trajs=trajs,
            obj_description=obj_name,
            visualize=False
        )
        
        regions = []
        for img in imgs:
            H, W = img.shape[:2]
            regions.append([0, 0, W, H])
        
        if self.crop:
            masked_frames, masks, trajs, regions = crop_images(masked_frames, masks, trajs)
        
        return masked_frames, masks, trajs, regions
    
    def clip_filtering(self, retrieved_data_dict, obj_prompt):
        query_frame = retrieved_data_dict['masked_query']
        retrieved_frames = retrieved_data_dict['masked_img']
        text_feature = self.clip_model.get_text_feature(obj_prompt)
        query_object = Image.fromarray(query_frame).convert('RGB')
        query_object_feature = self.clip_model.get_vision_feature(query_object)

        object_features = []
        for retrieved_frame in retrieved_frames:
            object_feature = self.clip_model.get_vision_feature(
                Image.fromarray(retrieved_frame).convert('RGB')
            )
            object_features.append(object_feature)

        visual_similarity = self.clip_model.compute_similarity(query_object_feature, torch.cat(object_features, dim=0))
        text_similarity = self.clip_model.compute_similarity(text_feature, torch.cat(object_features, dim=0))
        joint_similarity = visual_similarity * text_similarity

        sorted_index = joint_similarity.argsort()[0, ::-1]
        sorted_retrieved_data_dict = {
            "query_img": retrieved_data_dict["query_img"],
            "query_mask": retrieved_data_dict["query_mask"],
            "masked_query": retrieved_data_dict["masked_query"],
            "query_region": retrieved_data_dict["query_region"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": []
        }
        
        for idx in range(len(sorted_index)):
            curr = sorted_index[idx]
            prev = sorted_index[idx-1] if idx > 0 else 0
            if (retrieved_data_dict["mask"][curr].all()): # mask is all 1, meaning no mask
                continue
            if (self.clip_model.compute_similarity(object_features[curr], object_features[prev]) < 0.8 and joint_similarity[0, curr] > 0.1) or idx == 0:
                sorted_retrieved_data_dict["img"].append(retrieved_data_dict["img"][sorted_index[idx]])
                sorted_retrieved_data_dict["traj"].append(retrieved_data_dict["traj"][sorted_index[idx]])
                sorted_retrieved_data_dict["masked_img"].append(retrieved_data_dict['masked_img'][sorted_index[idx]])
                sorted_retrieved_data_dict["mask"].append(retrieved_data_dict['mask'][sorted_index[idx]])
                
                if len(sorted_retrieved_data_dict["img"]) >= MAX_IMD_RANKING_NUM:
                    break

        return sorted_retrieved_data_dict
    
    def visualize_top5(self, topk_retrieved_data_dict, save_name):
        img_pil_list = []
        img_pil_list.append(Image.fromarray(topk_retrieved_data_dict["masked_query"]).convert('RGB'))
        for img in topk_retrieved_data_dict["masked_img"]:
            img_pil_list.append(Image.fromarray(img).convert('RGB'))

        result_img = concat_images_with_lines(img_pil_list)
        result_img.save(os.path.join(self.save_root, save_name))
    
    def imd_ranking(self, sorted_retrieved_data_dict, obj_prompt):
        # 排序, 并且保留前 5 的结果
        src_ft = extract_ft(Image.fromarray(sorted_retrieved_data_dict['masked_query']).convert("RGB"), prompt=obj_prompt, ftype='sd') # 1,c,h,w
        src_mask = sorted_retrieved_data_dict["query_mask"]
        imd_distances = []
        sorted_retrieved_data_dict["feat"] = []
        for idx in tqdm(range(len(sorted_retrieved_data_dict["img"]))):
            tgt_ft = extract_ft(Image.fromarray(sorted_retrieved_data_dict['masked_img'][idx]).convert("RGB"), prompt=obj_prompt, ftype='sd')
            sorted_retrieved_data_dict["feat"].append(tgt_ft)
            tgt_mask = sorted_retrieved_data_dict["mask"][idx]
            imd_distances.append(get_distance_imd(src_ft, tgt_ft, src_mask, tgt_mask))
        sorted_index = np.argsort(imd_distances) # from smaller to larger, but the smaller, the better
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "query_region": sorted_retrieved_data_dict["query_region"],
            "query_feat": src_ft,
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "feat": []
        }
        for idx in range(self.topk):
            if idx >= len(sorted_index):
                break
            curr = sorted_index[idx]
            topk_retrieved_data_dict["img"].append(sorted_retrieved_data_dict["img"][curr])
            topk_retrieved_data_dict["traj"].append(sorted_retrieved_data_dict["traj"][curr])
            topk_retrieved_data_dict["masked_img"].append(sorted_retrieved_data_dict['masked_img'][curr])
            topk_retrieved_data_dict["mask"].append(sorted_retrieved_data_dict['mask'][curr])
            topk_retrieved_data_dict["feat"].append(sorted_retrieved_data_dict['feat'][curr])
        return topk_retrieved_data_dict

    def load_retrieved_task_from_pkl(self, retrieved_task):
        task_dir = os.path.join(self.subset_dir, self.data_source, retrieved_task.replace(" ", "_"))
        if os.path.exists(os.path.join(task_dir, retrieved_task.replace(" ", "_") + "_new.pkl")):
            with open(os.path.join(task_dir, retrieved_task.replace(" ", "_") + "_new.pkl"), 'rb') as f:
                retrieved_data_dict = pickle.load(f)
        return retrieved_data_dict

    def retrieve(self, current_task, current_obs, log=False, visualize=False):
        """
            current_task: "open the drawer"
            current_obs: numpy rgb image
        """
        if log: print("<1> Retrieve the most similar task")
        retrieved_task = self.language_retrieve(current_task)
        obj_name = retrieved_task.split(" ")[-1]
        obj_prompt = f"A photo of {obj_name}"
        if log: print(f"Retrieved task: {retrieved_task}")

        # retrieved_data_dict = load_retrieved_task(subset_dir, retrieved_task) # load from raw data
        if log: print("<2> Load the first frames and trajs of the episodes of the retrieved task")
        retrieved_data_dict = self.load_retrieved_task_from_pkl(retrieved_task) # img, traj, masked_img, mask

        if log: print("<3> Segment out the object from our observation")
        # TODO: 这里是不是需要从 current_task 中解析出物体的名称先
        query_obj_description = current_task.split(" ")[-1]
        # masked_frames, masks, trajs, regions
        query_frame, query_mask, _, query_region = self.segment_objects([current_obs], query_obj_description)
        
        retrieved_data_dict['query_img'] = current_obs
        retrieved_data_dict['query_mask'] = query_mask[0]
        retrieved_data_dict['masked_query'] = query_frame[0]
        retrieved_data_dict['query_region'] = query_region[0]
        
        if "masked_img" not in retrieved_data_dict.keys(): # not preprocessed
            assert True and "masked_img not in retrieved_data_dict.keys()"
            if log: print("<3.5> Retrieved data are not processed, processing...")
            masked_frames, frame_masks, trajs = self.segment_objects(retrieved_data_dict["img"], obj_name, retrieved_data_dict["traj"])
            retrieved_data_dict['masked_img'] = masked_frames
            retrieved_data_dict['mask'] = frame_masks
            retrieved_data_dict['traj'] = trajs

        if log: print("<4> Semantic filtering...")
        sorted_retrieved_data_dict = self.clip_filtering(retrieved_data_dict, obj_prompt)

        if log: print("<5> Geometrical retrieval...")
        topk_retrieved_data_dict = self.imd_ranking(sorted_retrieved_data_dict, obj_prompt)
        
        if visualize and False:
            # visualize topk_retrieved_data_dict
            import napari
            viewer = napari.Viewer()
            viewer.title = "retrieved reference data by RAM"
            viewer.add_image(topk_retrieved_data_dict["query_img"], name="query_img")
            viewer.add_image(topk_retrieved_data_dict["query_mask"] * 255, name="query_mask")
            viewer.add_image(topk_retrieved_data_dict["masked_query"], name="masked_query")
            
            for i in range(len(topk_retrieved_data_dict["img"])):
                viewer.add_image(topk_retrieved_data_dict["img"][i], name=f"ref_img_{i}")
                masked_img = topk_retrieved_data_dict["masked_img"][i]
                masked_img = np.array(draw_points_on_image(masked_img, [topk_retrieved_data_dict["traj"][i][0]], 5))
                viewer.add_image(masked_img, name=f"masked_ref_img_{i}")
                viewer.add_image(topk_retrieved_data_dict["mask"][i] * 255, name=f"ref_img_mask_{i}")
            napari.run()
        
        """
        这里的 query_mask 和 masked_query 都是 cropped 过的
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "query_region": sorted_retrieved_data_dict["query_region"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": []
        }
        """
        return topk_retrieved_data_dict
