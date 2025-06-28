import numpy as np
import pandas as pd

from pyntcloud import PyntCloud

def save_pc_to_ply(points, colors, save_root = "outout", save_name = "output.ply"):
    
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points, colors)),
        columns=["x", "y", "z", "red", "green", "blue"]))

    cloud.to_file(save_root +  "/" + save_name)
    
