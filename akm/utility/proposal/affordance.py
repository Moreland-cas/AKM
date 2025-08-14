import cv2
import numpy as np
from PIL import Image
from matplotlib import cm
from akm.utility.utils import (
    draw_points_on_image,
    concatenate_images
)


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
            H, W, 3: The locations of the contact points to be sampled on this RGB image
        cos_map:
            cropped_H, cropped_W, values between (-1, 1)
        cropped_mask:
            cropped_H, cropped_W, bool type
        Represents the segmentation mask of the object in the cropped region
        cropped_region:
            (x_min, y_min, x_max, y_max)
        Represents the position of the cropped_region in the original RGB_img
        NOTE: The size of the cos_map is different from the cropped mask/region 
        """
        self.rgb_img = rgb_img
        self.cos_map = cos_map # H, W, in range [-1, 1]
        self.H, self.W = rgb_img.shape[:2]
        self.cropped_H, self.cropped_W = cropped_mask.shape[:2]
        
        self.cropped_mask = cropped_mask
        self.cropped_region = cropped_region

    def get_colored_cos_map(self):
        # cmap = cm.get_cmap("jet")
        cmap = cm.get_cmap("viridis")
        tmp_cos_map = np.clip(self.cos_map, -1, 1)
        prob_map = (tmp_cos_map + 1) / 2
        colored_image = cmap(prob_map)[..., :3]  # Returns (H, W, 4) RGBA array
        # Convert to 8-bit RGB (ignore the alpha channel)
        colored_image = (colored_image * 255).astype(np.uint8)
        colored_image = Image.fromarray(colored_image, mode="RGB")
        return colored_image
        
    def visualize(self):
        image = self.get_colored_cos_map()
        image.show()
    
    def rgb_to_cos_frame(self, u_rgb, v_rgb):
        """
        Convert (u, v) in rgb_img to (u, v) in cos_map
        (u_rgb, v_rgb) -> (normalized_u, normalized_v) -> (u_cos, v_cos) 
        """
        normalized_u = (u_rgb - self.cropped_region[0]) / self.cropped_W
        normalized_v = (v_rgb - self.cropped_region[1]) / self.cropped_H
        
        u_cos = normalized_u * self.cos_map.shape[1]
        v_cos = normalized_v * self.cos_map.shape[0]
        
        return (u_cos, v_cos)
    
    def cos_to_rgb_frame(self, u_cos, v_cos):
        """
        Convert (u, v) in cos_map to (u, v) in rgb_img
        (u_cos, v_cos) -> (normalized_u, normalized_v) -> (u_rgb, v_rgb)
        """
        normalized_u = u_cos/ self.cos_map.shape[1]
        normalized_v = v_cos / self.cos_map.shape[1]
        
        u_rgb = self.cropped_region[0] + normalized_u * self.cropped_W
        v_rgb = self.cropped_region[1] + normalized_v * self.cropped_H
        
        return (u_rgb, v_rgb)
    
    def get_obj_mask(self, visualize=False):
        # Return a mask of the same size as rgb_img, where the cropped_region area is filled with cropped_mask
        obj_mask = np.zeros(self.rgb_img.shape[:2]).astype(np.bool_) # H, W
        obj_mask[self.cropped_region[1]:self.cropped_region[3], self.cropped_region[0]:self.cropped_region[2]] = self.cropped_mask
        
        if visualize:
            Image.fromarray(obj_mask).show()
        return obj_mask
    
    def mask_cos_map(self):
        """
        Set the value of points in cos_map that are not in cropped_mask to -1 
        """
        # First, you need to scale the cropped_mask to the size of cos_map (interpolation method)
        mask_uint8 = self.cropped_mask.astype(np.uint8) 

        # Use cv2.resize for nearest neighbor interpolation
        cos_w = self.cos_map.shape[1]
        cos_h = self.cos_map.shape[0]
        resized_mask = cv2.resize(mask_uint8, (cos_w, cos_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask > 0
        self.cos_map[~resized_mask] = -1e6
        
    def uninit_cosmap(self):
        """
        Set the values of points in cos_map that are not in cropped_mask to -1, and those in cropped_mask to 1.  
        """
        self.mask_cos_map()
        # First, you need to scale the cropped_mask to the size of cos_map (interpolation method)
        mask_uint8 = self.cropped_mask.astype(np.uint8)  

        cos_w = self.cos_map.shape[1]
        cos_h = self.cos_map.shape[0]
        resized_mask = cv2.resize(mask_uint8, (cos_w, cos_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask > 0
        self.cos_map[resized_mask] = 1
        
    def sample_prob(self, alpha=10, num_samples=1, return_rgb_frame=True, visualize=False):
        """
        First, get the prob_map based on the cos_map, then randomly sample a point and ensure that it falls within the cropped_mask  
        """
        self.mask_cos_map() 
        prob_map = (self.cos_map + 1.) / 2 
        prob_map_scaled = prob_map * np.exp(prob_map * alpha)
        prob_map_normalized = prob_map_scaled / prob_map_scaled.sum()
        self.prob_map = prob_map_normalized
        
        flat_prob_map = prob_map_normalized.flatten()
        indices = np.random.choice(np.arange(flat_prob_map.size), size=num_samples, p=flat_prob_map)
        
        v_cos, u_cos = np.unravel_index(indices, self.cos_map.shape)
        u_rgb, v_rgb = zip(*[self.cos_to_rgb_frame(u, v) for u, v in zip(u_cos, v_cos)])
        
        # Since the size of cos_map/prob_map is scaled and does not strictly correspond to the pixel coordinates of cropped_region,
        # the points sampled from prob_map should be normalized
        if visualize:
            image_cos = self.get_colored_cos_map()
            image_cos = draw_points_on_image(
                image=image_cos,
                uv_list=list(zip(u_cos, v_cos)),
                radius=1,
                normalized_uv=False
            )
            
            image_rgb = draw_points_on_image(
                image=self.rgb_img,
                uv_list=list(zip(u_rgb, v_rgb)),
                radius=1,
                normalized_uv=False
            )
            concatenate_images(image_cos, image_rgb).show()
        
        if return_rgb_frame:
            return np.array(list(zip(u_rgb, v_rgb)))   
        else:
            return np.array(list(zip(u_cos, v_cos)))
    
    def sample_highest(self, visualize=False):
        """
        Sample the point with the maximum value on cos_map and ensure that the point falls within the cropped_mask
        """
        self.mask_cos_map()
        max_index = np.unravel_index(np.argmax(self.cos_map), self.cos_map.shape)
        v_cos, u_cos = max_index
        u_rgb, v_rgb = self.cos_to_rgb_frame(u_cos, v_cos)
        
        # Since the size of cos_map/prob_map is scaled and does not strictly correspond to the pixel coordinates of cropped_region, 
        # the points sampled from prob_map should be normalized
        if visualize:
            image_cos = self.get_colored_cos_map()
            image_cos = draw_points_on_image(
                image=image_cos,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )
            image_rgb = draw_points_on_image(
                image=self.rgb_img,
                uv_list=[(u_rgb, v_rgb)],
                radius=5,
                normalized_uv=False
            )
            concatenate_images(image_cos, image_rgb).show()
        return np.array([u_rgb, v_rgb])  
       
    def update(self, neg_uv_rgb, update_sigma=0.05, visualize=False):
        """
        neg_uv_rgb:
        Failed attempt point (u_rgb, v_rgb)
        Update self.cos_map based on neg_uv_rgb. Since the value of cos_map is in the range (-1, 1), this range must be considered when updating.
        """
        assert neg_uv_rgb is not None, "neg_uv_rgb cannot be zero"
        # Convert failed RGB coordinates to cos_map coordinates
        u_cos, v_cos = self.rgb_to_cos_frame(neg_uv_rgb[0], neg_uv_rgb[1])

        if visualize:
            image_cos_old = self.get_colored_cos_map()
            image_cos_old = draw_points_on_image(
                image=image_cos_old,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )

        # Define the Gaussian kernel standard deviation for full image update
        sigma = int(self.cos_map.shape[0] * update_sigma)

        # Generate coordinate grid
        u_indices = np.arange(self.cos_map.shape[1])
        v_indices = np.arange(self.cos_map.shape[0])
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)

        # compute distance
        distances = np.sqrt((u_grid - u_cos) ** 2 + (v_grid - v_cos) ** 2)

        # Calculate weights using Gaussian function
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # Update cos_map, using weights to reduce values
        self.cos_map -= weights * 0.5  

        if visualize:
            image_cos_new = self.get_colored_cos_map()
            u_rgb, v_rgb = self.sample_highest()
            u_cos, v_cos = self.rgb_to_cos_frame(u_rgb, v_rgb)
            image_cos_new = draw_points_on_image(
                image=image_cos_new,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )
            concatenate_images(image_cos_old, image_cos_new).show()
