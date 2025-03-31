import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
from embodied_analogy.utility.utils import (
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
                H, W, 3 要在这个 rgb 图上采样接触点的位置
            cos_map:
                cropped_H, cropped_W, 值在 (-1, 1) 之间
            cropped_mask: 
                cropped_H, cropped_W, boolen type
                代表 cropped region 中物体的分割 mask
            cropped_region:
                (x_min, y_min, x_max, y_max)
                代表 cropped_region 在原始的 rgb_img 中的位置
            NOTE: cos_map 的大小与 cropped mask/region 并不一样
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
        prob_map = (self.cos_map + 1) / 2
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
            将 rgb_img 中的 (u, v) 转换为 cos_map 中的 (u, v)
            (u_rgb, v_rgb) -> (normalized_u, normalized_v) ->  (u_cos, v_cos)
        """
        normalized_u = (u_rgb - self.cropped_region[0]) / self.cropped_W
        normalized_v = (v_rgb - self.cropped_region[1]) / self.cropped_H
        
        u_cos = normalized_u * self.cos_map.shape[1]
        v_cos = normalized_v * self.cos_map.shape[0]
        
        return (u_cos, v_cos)
    
    def cos_to_rgb_frame(self, u_cos, v_cos):
        """
            将 cos_map 中的 (u, v) 转换为 rgb_img 中的 (u, v)
            (u_cos, v_cos) -> (normalized_u, normalized_v) -> (u_rgb, v_rgb)
        """
        normalized_u = u_cos/ self.cos_map.shape[1]
        normalized_v = v_cos / self.cos_map.shape[1]
        
        u_rgb = self.cropped_region[0] + normalized_u * self.cropped_W
        v_rgb = self.cropped_region[1] + normalized_v * self.cropped_H
        
        return (u_rgb, v_rgb)
    
    def get_obj_mask(self, visualize=False):
        # 返回一个大小与 rgb_img 一样的 mask, 其中 cropped_region 区域用 cropped_mask 填充
        obj_mask = np.zeros(self.rgb_img.shape[:2]).astype(np.bool_) # H, W
        obj_mask[self.cropped_region[1]:self.cropped_region[3], self.cropped_region[0]:self.cropped_region[2]] = self.cropped_mask
        
        if visualize:
            Image.fromarray(obj_mask).show()
        return obj_mask
    
    def mask_cos_map(self):
        """
            将 cos_map 中不在 cropped_mask 中的点值设置为 -1
        """
        # 首先需要将 cropped_mask 缩放到 cos_map 大小（插值的方式）
        mask_uint8 = self.cropped_mask.astype(np.uint8)  # 将 True 转换为 255，False 转换为 0

        # 使用cv2.resize进行最近邻插值
        cos_w = self.cos_map.shape[1]
        cos_h = self.cos_map.shape[0]
        resized_mask = cv2.resize(mask_uint8, (cos_w, cos_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask > 0
        self.cos_map[~resized_mask] = -1
        
    def sample_prob(self, alpha=10, visualize=False):
        """
            首先根据 cos_map 得到 prob_map, 然后随机 sample 一个, 并且保证该点落在 cropped_mask 中
            TODO Not working well, might need to be fixed
        """
        if self.cos_map.max() == -1:
            assert False, "cos_map 中没有值"
            
        self.mask_cos_map() # 把 mask 外的部分的 cos sim 变为 -1
        prob_map = (self.cos_map + 1.) / 2 # 值域变为 (0, 1)
        prob_map_scaled = prob_map * np.exp(prob_map * alpha)
        prob_map_normalized = prob_map_scaled / prob_map_scaled.sum()
        self.prob_map = prob_map_normalized
        
        flat_prob_map = prob_map_normalized.flatten()
        index = np.random.choice(np.arange(flat_prob_map.size), p=flat_prob_map)
        u_cos, v_cos = divmod(index, self.cos_map.shape[1]) 
        u_rgb, v_rgb = self.cos_to_rgb_frame(u_cos, v_cos)
        
        # 由于 cos_map/prob_map 的大小是经过缩放的, 和 cropped_region 的像素坐标并不严格对应, 因此从 prob_map 中采样出的点应该是经过归一化的
        if visualize:
            image_cos = self.get_colored_cos_map()
            image_cos = draw_points_on_image(
                image=image_cos,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )
            # image_cos.show()
            
            image_rgb = draw_points_on_image(
                image=self.rgb_img,
                uv_list=[(u_rgb, v_rgb)],
                radius=5,
                normalized_uv=False
            )
            # image_rgb.show()
            
            # image_mask = draw_points_on_image(
            #     image=Image.fromarray(self.cropped_mask * 255.).convert("RGB"),
            #     uv_list=[(u, v)],
            #     radius=5,
            #     normalized_uv=True
            # )
            # image_mask.show()
            
            concatenate_images(image_cos, image_rgb).show()
            
        return (u_rgb, v_rgb)   
    
    def sample_highest(self, visualize=False):
        """
            在 cos_map 上 sample 出值最大的点, 并且保证该点落在 cropped_mask 中
        """
        if self.cos_map.max() == -1:
            assert False, "cos_map 中没有值"
            
        self.mask_cos_map()
        max_index = np.unravel_index(np.argmax(self.cos_map), self.cos_map.shape)
        # max_index 是 (v, u) 的形式
        v_cos, u_cos = max_index
        u_rgb, v_rgb = self.cos_to_rgb_frame(u_cos, v_cos)
        
        # 由于 cos_map/prob_map 的大小是经过缩放的, 和 cropped_region 的像素坐标并不严格对应, 因此从 prob_map 中采样出的点应该是经过归一化的
        if visualize:
            image_cos = self.get_colored_cos_map()
            image_cos = draw_points_on_image(
                image=image_cos,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )
            # image_cos.show()
            
            image_rgb = draw_points_on_image(
                image=self.rgb_img,
                uv_list=[(u_rgb, v_rgb)],
                radius=5,
                normalized_uv=False
            )
            # image_rgb.show()
            
            # image_mask = draw_points_on_image(
            #     image=Image.fromarray(self.cropped_mask * 255.).convert("RGB"),
            #     uv_list=[(u, v)],
            #     radius=5,
            #     normalized_uv=True
            # )
            # image_mask.show()
            
            concatenate_images(image_cos, image_rgb).show()
        return np.array([u_rgb, v_rgb])  
       
    def update(self, neg_uv_rgb, visualize=False):
        """
        neg_uv_rgb: 
            失败的尝试点 (u_rgb, v_rgb) 
        根据 neg_uv_rgb 来更新 self.cos_map, 由于 cos_map 的值在 (-1, 1), 所以更新时要考虑这个值域
        """
        assert neg_uv_rgb is not None, "neg_uv_rgb 不能为空"
        # 将失败的 RGB 坐标转换为 cos_map 坐标
        u_cos, v_cos = self.rgb_to_cos_frame(neg_uv_rgb[0], neg_uv_rgb[1])

        if visualize:
            image_cos_old = self.get_colored_cos_map()
            image_cos_old = draw_points_on_image(
                image=image_cos_old,
                uv_list=[(u_cos, v_cos)],
                radius=5,
                normalized_uv=False
            )

        # 定义全图更新的高斯核标准差
        sigma = int(self.cos_map.shape[0] * 0.1)

        # 生成坐标网格
        u_indices = np.arange(self.cos_map.shape[1])
        v_indices = np.arange(self.cos_map.shape[0])
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)

        # 计算距离
        distances = np.sqrt((u_grid - u_cos) ** 2 + (v_grid - v_cos) ** 2)

        # 使用高斯函数计算权重
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # 更新 cos_map，使用权重降低值
        self.cos_map -= weights * 0.5  
        self.cos_map = np.clip(self.cos_map, -1, None)  # 确保不低于 -1

        if visualize:
            # 可视化更新后的 cos_map
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

class Affordance_map_3d(Affordance_map_2d):
    def __init__(self, rgb_img, cos_map, cropped_mask, cropped_region):
        super().__init__(rgb_img, cos_map, cropped_mask, cropped_region)
        
    def sample_grasp(self):
        # sample 一个 grasp 和 post_grasp dir
        pass 

if __name__ == "__main__":
    input_data = np.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/ram_proposal/affordance_map_2d_input.npz")
    # 测试一下 Affordance_map_2d
    affordance_map_2d = Affordance_map_2d(
        rgb_img=input_data["rgb_img"],
        cos_map=input_data["cos_map"],
        cropped_mask=input_data["cropped_mask"],
        cropped_region=input_data["cropped_region"],
    )
    affordance_map_2d.get_obj_mask(False)
    # uv_rgb = affordance_map_2d.sample_prob(alpha=0, visualize=True)
    uv_rgb = affordance_map_2d.sample_highest(visualize=True)
    affordance_map_2d.update(uv_rgb, visualize=False)
    uv_rgb = affordance_map_2d.sample_highest(visualize=True)
    affordance_map_2d.update(uv_rgb, visualize=False)
    uv_rgb = affordance_map_2d.sample_highest(visualize=True)
    