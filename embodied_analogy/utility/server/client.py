import requests
def run_anygrasp_remotely(
    points_input,
    colors, 
    lims
):
    # 服务端的地址和端口
    server_url = "http://10.1.100.36:5000/api/calculate"

    # 要发送的数据
    data = {
        "points_input": points_input,
        "colors": colors, 
        "lims": lims
    }

    # 发送 POST 请求
    response = requests.post(server_url, json=data)
    result = response.json()
    return result["gg"]

if __name__ == "__main__":
    from embodied_analogy.representation.obj_repr import Obj_repr
    from embodied_analogy.representation.basic_structure import Frame
    import numpy as np
    o: Obj_repr = Obj_repr.load("/data/zby_data/assets/logs/explore_56/40147_1_prismatic/obj_repr.npy")
    o.frames[0]: Frame
    pcs, colors = o.frames[0].get_env_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=True,
        visualize=False
    )
    gg = run_anygrasp_remotely(
        points_input=pcs,
        colors=colors, 
        lims=np.array([-1, 1, -1, 1, -1, 1]) * 10
    )
    print(len(gg), gg[0])