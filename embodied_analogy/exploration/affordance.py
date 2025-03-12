import numpy as np
from PIL import Image
from matplotlib import cm

class Affordance_map_2d:
    def __init__(
        self, 
        rgb_img,
        cos_map, 
        cropped_mask,
        cropped_region,
        # alpha: float=20
    ):
        """
            rgb_img: 
                H, W, 3 要在这个 rgb 图上采样接触点的位置
            cos_map:
                cropped_H, cropped_W, 值在 (-1, 1) 之间
            cropped_mask: 
                cropped_H, cropped_W, boolen type
                代表 cropped region 中物体的分割 mask
            cropped_region:
                (x_min, y_min, x_max, y_max)
                代表 cropped_region 在原始的 rgb_img 中的位置
        """
        self.cos_map = cos_map # H, W, in range [-1, 1]
        self.H, self.W = rgb_img.shape[:2]
        self.cropped_H, self.cropped_W = cropped_mask.shape[:2]
        
        self.cropped_mask = cropped_mask
        self.cropped_region = cropped_region

    def visualize(self):
        # pil image for visualization
        # cmap = cm.get_cmap("jet")
        cmap = cm.get_cmap("viridis")
        colored_image = cmap(self.similarity_map.cpu().numpy())  # Returns (H, W, 4) RGBA array
        # Convert to 8-bit RGB (ignore the alpha channel)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        # Convert to PIL Image
        Image.fromarray(colored_image, mode="RGB").show()
    
    def sample_highest(self, visualize=False):
        max_index = np.unravel_index(np.argmax(self.cos_map), self.cos_map.shape)
        # max_index 是 (v, u) 的形式
        v, u = max_index
        return (u, v)
    
    def update(self, negative_point, visualize=False):
        """
        negative_point: 
            (u, v) 失败的尝试点
        根据 negative_point 来更新 self.cos_map
        """
        pass
    
    def sample_deprecated(self, num_samples=50, visualize=True):
        # Sample indices based on the probability distribution
        sampled_indices = np.random.choice(len(self.prob_np), size=num_samples, p=self.prob_np, replace=False)
        # Convert flat indices to 2D coordinates (y, x)
        y_coords, x_coords = np.unravel_index(sampled_indices, (self.H, self.W))

        # Normalize coordinates to range [0, 1]
        u_coords = x_coords / self.W
        v_coords = y_coords / self.H

        # Combine u and v into a single array of shape (num_samples, 2)
        sampled_coordinates = np.stack((u_coords, v_coords), axis=-1)
        
        if visualize:
            img = draw_points_on_image(self.pil_image, sampled_coordinates, radius=3)
            img.show()
            
        return sampled_coordinates
    

class Affordance_map_3d(Affordance_map_2d):
    def __init__(self, rgb_img, cos_map, cropped_mask, cropped_region):
        super().__init__(rgb_img, cos_map, cropped_mask, cropped_region)
        
    def sample_grasp(self):
        # sample 一个 grasp 和 post_grasp dir
        pass 
    