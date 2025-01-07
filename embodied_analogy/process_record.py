import numpy as np
from embodied_analogy.utils import dot_on_image, pil_images_to_mp4
from PIL import Image

class RecordDataReader():
    def __init__(self, record_path_prefix, file_name) -> None:
        self.record_path_prefix = record_path_prefix
        self.filename = file_name
        self.data = np.load(record_path_prefix + file_name, allow_pickle=True)
        
    def process_data(self):
        # 获取 first contact point、panda_hand 的轨迹、rgbd数据
        
        # 读取 after_contact 之后的first contact point
        for data_dict in self.data["traj"]:
            after_close = data_dict["after_close"]
            if not after_close:
                continue
            cp_2d = data_dict["contact_points_2d"] # N, 2
            if len(cp_2d) > 0:
                cp_2d_mean = cp_2d.mean(0) # 2
                self.first_cp_2d = cp_2d_mean
                break
        
        # 提取 panda_hand 的轨迹
        panda_hand_pos = []
        panda_hand_quat = []
        for data_dict in self.data["traj"]:
            after_close = data_dict["after_close"]
            if not after_close:
                continue
            panda_hand_pos.append(data_dict["panda_hand_pos"])
            panda_hand_quat.append(data_dict["panda_hand_quat"])
            
        self.panda_hand_pos = np.array(panda_hand_pos)
        # save relative pose
        self.panda_hand_pos -= panda_hand_pos[0]
        self.panda_hand_quat = np.array(panda_hand_quat)
        
        # 提取 rgbd 数据
        rgb = []
        depth = []
        for data_dict in self.data["traj"]:
            after_close = data_dict["after_close"]
            if not after_close:
                continue
            rgb.append(data_dict["rgb_np"])
            depth.append(data_dict["depth_np"])
            
        self.rgb = np.stack(rgb) # numpy array in shape [T, H, W, 3], in uint8
        self.depth = np.stack(depth)[..., None] # numpy array in shape [T, H, W, 1], in meters
        self.seg = (self.data["object_seg"] != 0) & (self.data["object_seg"] != 1) # numpy array in shape [H, W]
        
    def get_first_view_img(self, idx=0):
        # return pil image of the first view
        pil_img =  Image.fromarray(self.data["traj"][idx]["rgb_np"])
        return pil_img
    
    def get_object_image(self):
        # return object image in pil format, without franka arm
        return self.data["object_image"]
    
    def visualize_contact_as_video(self):
        rgb_pil_with_contact = []
        for data_dict in self.data["traj"]:
            after_close = data_dict["after_close"]
            # if not after_close:
            #     continue
            rgb_np = data_dict["rgb_np"]
            rgb_pil = Image.fromarray(rgb_np)
            cp_2d = data_dict["contact_points_2d"] # N, 2
            if len(cp_2d) > 0:
                cp_2d_mean = cp_2d.mean(0) # 2
                rgb_pil = dot_on_image(rgb_pil, [cp_2d_mean], radius=3)
            rgb_pil_with_contact.append(rgb_pil)
            
        save_name = self.filename.split(".")[0]
        pil_images_to_mp4(rgb_pil_with_contact, self.record_path_prefix + f"/{save_name}.mp4")
        
        
if __name__ == "__main__":
    record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
    file_name = "/2025-01-07_18-06-10.npz"
    dr = RecordDataReader(record_path_prefix, file_name)
    dr.process_data()
    dr.visualize_contact_as_video()
