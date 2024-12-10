import numpy as np
from embodied_analogy.utils import draw_red_dot, pil_images_to_mp4
from PIL import Image

class DataReader():
    def __init__(self, recorded_data_path, file_name) -> None:
        self.recorded_data_path = recorded_data_path
        self.filename = file_name
        self.data = np.load(recorded_data_path + file_name, allow_pickle=True)
    def process_data(self):
        # 读取 after_contact 之后的数据
        pass
    def visualize_contact_as_video(self):
        rgb_pil_with_contact = []
        for data_dict in self.data["traj"]:
            after_close = data_dict["after_close"]
            if not after_close:
                continue
            rgb_np = data_dict["rgb_np"]
            rgb_pil = Image.fromarray(rgb_np)
            cp_2d = data_dict["contact_points_2d"] # N, 2
            if len(cp_2d) > 0:
                cp_2d_mean = cp_2d.mean(0) # 2
                rgb_pil = draw_red_dot(rgb_pil, cp_2d_mean[0], cp_2d_mean[1], radius=3)
            rgb_pil_with_contact.append(rgb_pil)
            
        save_name = self.filename.split(".")[0]
        pil_images_to_mp4(rgb_pil_with_contact, self.recorded_data_path + f"/{save_name}.mp4")
        
if __name__ == "__main__":
    recorded_data_path = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
    file_name = "/recorded_data_2.npz"
    dr = DataReader(recorded_data_path, file_name)
    dr.visualize_contact_as_video()