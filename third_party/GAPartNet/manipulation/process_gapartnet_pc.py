import glob
import open3d
import numpy as np


paths = glob.glob('/home/haoran/Projects/Part/GAPartNet/gym/gapartnet_obj/*-articulated-point_cloud.ply')
dict = {}
for path in paths:
    # read ply points
    points = open3d.io.read_point_cloud(path)
    xyz = np.asarray(points.points)
    min_z = xyz.min(axis=0)[2]
    # import pdb; pdb.set_trace()
    id = path.split('/')[-1].split('-')[0]
    dict[id] = min_z

import json
# write
with open('gapartnet_obj_min_z.json', 'w') as f:
    json.dump(dict, f) 