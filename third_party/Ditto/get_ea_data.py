import os
import numpy as np
from akm.representation.basic_structure import Frame
from akm.representation.obj_repr import Obj_repr

# obj_folder = "/home/zby/Programs/AKM/asset_book/logs/explore_424/40147_1_prismatic"
# obj_folder = "/home/zby/Programs/AKM/asset_book/logs/explore_424/45162_0_revolute"
obj_folder = "/home/zby/Programs/AKM/asset_book/logs/explore_424/45168_0_prismatic"
obj_repr: Obj_repr = Obj_repr.load(os.path.join(obj_folder, "obj_repr.npy"))

# 提取首尾两帧
frame_start: Frame = obj_repr.frames[0]
frame_start.segment_obj(
    obj_description=obj_repr.obj_description,
    post_process_mask=True,
    filter=True
)
pc_start, _ = frame_start.get_obj_pc(
    use_robot_mask=True, 
    use_height_filter=True,
    world_frame=True,
)

frame_end: Frame = obj_repr.frames[-1]
frame_end.segment_obj(
    obj_description=obj_repr.obj_description,
    post_process_mask=True,
    filter=True,
)
pc_end, _ = frame_end.get_obj_pc(
    use_robot_mask=True, 
    use_height_filter=True,
    world_frame=True,
    visualize=True,
)

# 保存
save_folder = "/home/zby/Programs/Ditto/data/custom_data"
np.save(
    os.path.join(save_folder, "pc_start.npy"),
    pc_start
)
np.save(
    os.path.join(save_folder, "pc_end.npy"),
    pc_end
)