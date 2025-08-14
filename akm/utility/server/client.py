import requests
import numpy as np
from graspnetAPI import GraspGroup


def run_anygrasp_remotely(
    points_input, # np.array
    colors, 
    lims
):
    server_url = "http://10.1.100.36:5000/api/calculate"
    # Server address and port
    data = {
        "points_input": points_input.tolist(),
        "colors": colors.tolist(), 
        "lims": lims.tolist()
    }

    # Sending a POST request
    response = requests.post(server_url, json=data)
    result = response.json()
    gg_array = result["gg_array"]
    if gg_array is None:
        return None
    else:
        gg = GraspGroup()
        gg.grasp_group_array = np.array(gg_array)
        return gg