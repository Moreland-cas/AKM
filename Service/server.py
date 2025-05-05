from flask import Flask, request, jsonify

app = Flask(__name__)

# 定义函数 y = f(x) = x^2
def f(x):
    return x ** 2

# 定义API接口
@app.route('/api/calculate', methods=['POST'])
def calculate():
    # 获取请求中的数据
    data = request.json
    if 'x' not in data:
        return jsonify({"error": "Missing 'x' parameter"}), 400

    x = data['x']
    if not isinstance(x, (int, float)):
        return jsonify({"error": "'x' must be a number"}), 400

    # 计算 y = f(x)
    y = f(x)
    response = {
        "x": x,
        "y": y
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 0.0.0.0 表示允许外部访问
