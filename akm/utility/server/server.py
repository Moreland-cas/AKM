import numpy as np
from flask import Flask, request, jsonify
from akm.utility.grasp.anygrasp import prepare_any_grasp_model


def run_anygrasp_helper(points_input, colors, lims):
    points_input = points_input.astype(np.float32)
    colors = colors.astype(np.float32)
    model = prepare_any_grasp_model(
        asset_path="/home/Programs/AKM/assets/ckpts/anygrasp/checkpoint_detection.tar"
    )
    gg, _ = model.get_grasp(
        points_input,
        colors, 
        lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )
    return gg

app = Flask(__name__)

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    # list
    points_input = data["points_input"]
    colors = data["colors"]
    lims = data["lims"]
    
    # to np.array
    points_input = np.array(points_input)
    colors = np.array(colors)
    lims = np.array(lims)
    
    gg = run_anygrasp_helper(points_input, colors, lims)
    if gg is None or len(gg) == 0:
        response = {
            "gg_array": None
        }
    else:
        gg_array = gg.grasp_group_array
        response = {
            "gg_array": gg_array.tolist()
        }
    return jsonify(response)


if __name__ == '__main__':
    # 0.0.0.0 means external access is allowed
    app.run(host='0.0.0.0', port=5000) 
