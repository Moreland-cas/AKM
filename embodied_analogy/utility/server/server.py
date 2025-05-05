from flask import Flask, request, jsonify
from embodied_analogy.utility.grasp.anygrasp import prepare_any_grasp_model
# 定义函数 y = f(x) = x^2
def run_anygrasp_helper(points_input, colors, lims):
    model = prepare_any_grasp_model()
    gg, _ = model.get_grasp(
        points_input,
        colors, 
        lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )
    return gg

# 定义API接口
@app.route('/api/calculate', methods=['POST'])
def calculate():
    # 获取请求中的数据
    data = request.json
    
    points_input = data["points_input"]
    colors = data["colors"]
    lims = data["lims"]
    
    gg = run_anygrasp_helper(points_input, colors, lims)
    response = {
        "gg": gg
    }
    return jsonify(response)

if __name__ == '__main__':
    app = Flask(__name__)
    app.run(host='0.0.0.0', port=5000)  # 0.0.0.0 表示允许外部访问
