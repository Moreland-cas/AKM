import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import threading
import time

# 创建 Dash 应用
app = dash.Dash(__name__)

# 初始化点云数据和帧索引
point_clouds = []
current_frame = 0

# 生成点云数据的函数
def generate_point_cloud(num_points, num_frames):
    global point_clouds
    for i in range(num_frames):
        # 随机生成点云数据
        x = np.random.rand(num_points)
        y = np.random.rand(num_points)
        z = np.random.rand(num_points)
        point_clouds.append((x, y, z))
        time.sleep(2)  # 模拟生成点云的时间

# 启动生成点云数据的线程
threading.Thread(target=generate_point_cloud, args=(100, 3), daemon=True).start()

# 创建 Dash 布局
app.layout = html.Div([
    html.Div(id='graphs-container'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 每1000毫秒（1秒）检查一次
        n_intervals=0
    ),
    html.Div(id='status')
])

# 回调函数更新图表
@app.callback(
    Output('graphs-container', 'children'),
    Output('status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global current_frame
    children = []

    if current_frame < len(point_clouds):
        # 获取当前帧的数据
        x, y, z = point_clouds[current_frame]

        # 创建散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5)
        )])

        # 设置图形布局
        fig.update_layout(
            title=f'Point Cloud Data - Frame {current_frame + 1}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=400
        )

        # 将图表添加到 children 列表
        children.append(html.Div([
            dcc.Graph(figure=fig),
            html.Hr()
        ]))

        # 更新帧索引
        current_frame += 1
        status_text = f'Generating frame {current_frame}...'
    else:
        status_text = 'All frames generated.'

    return children, status_text

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
