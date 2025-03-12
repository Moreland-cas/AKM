# import sapien.core as sapien
# import numpy as np
# from sapien.utils.viewer import Viewer

# engine = sapien.Engine()
# renderer = sapien.SapienRenderer()
# engine.set_renderer(renderer)

# scene_config = sapien.SceneConfig()
# scene = engine.create_scene(scene_config)

# scene.set_timestep(1 / 250.0)
# scene.add_ground(0)
# # physical_material = scene.create_physical_material(1, 1, 0.0)
# # scene.default_physical_material = physical_material

# scene.set_ambient_light([0.5, 0.5, 0.5])
# scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
# scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
# scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
# scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

# viewer = Viewer(renderer)
# viewer.set_scene(scene)
# # 创建默认Viewer
# # viewer = scene.create_viewer()

# # 默认分辨率
# # print("Default resolution: ", viewer.resolution)  # 输出类似 (640, 480)

# # 默认视场角 (FOV)
# # print("Default FOV (in degrees): ", np.rad2deg(viewer.))  # 默认是 45° 

# # # 默认近平面和远平面
# # print("Near plane: ", viewer.get_near())
# # print("Far plane: ", viewer.get_far())
# while not viewer.closed:
#     cur_time = scene.get_simulation_time()
#     print(cur_time)
#     scene.step()
#     scene.update_render()
#     viewer.render() 
    
# import numpy as np

# # 给定的变换矩阵（3x4）
# transform = np.array([
#     [-4.8380852e-01, -8.7517387e-01,  5.2154064e-07,  4.5627761e-01],
#     [-1.6098604e-01,  8.8994779e-02, -9.8293614e-01,  3.8961518e-01],
#     [ 8.6023998e-01, -4.7555298e-01, -1.8394721e-01,  8.8079190e-01]
# ], dtype=np.float32)

# # 扩展为齐次变换矩阵 (4x4)
# T = np.vstack([transform, np.array([0, 0, 0, 1])])

# # 提取旋转矩阵 R 和平移向量 t
# R = T[:3, :3]
# t = T[:3, 3]

# # 计算逆矩阵
# R_inv = R.T
# t_inv = -np.dot(R_inv, t)

# # 组装逆矩阵
# T_inv = np.eye(4)
# T_inv[:3, :3] = R_inv
# T_inv[:3, 3] = t_inv

# print("Inverse Transformation Matrix:")
# print(T_inv)

import napari
from PIL import Image
import numpy as np

from embodied_analogy.utility.utils import draw_points_on_image
topk_retrieved_data_dict = np.load("/home/zby/Programs/Embodied_Analogy/data.npy", allow_pickle=True).item()

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
