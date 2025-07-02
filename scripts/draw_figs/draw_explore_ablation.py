import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('svg')
import numpy as np

plt.rcParams.update({
    'font.size': 18,          # 全局默认字体大小
    # 'axes.titlesize': 16,     # 标题字体大小
    # 'axes.labelsize': 14,     # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,    # x轴刻度字体大小
    # 'ytick.labelsize': 12,    # y轴刻度字体大小
    # 'legend.fontsize': 12     # 图例字体大小
})

def plot_comparison(data_dict, title="Method Comparison", xlabel="Number of Interactions (times)", ylabel="Explore Success Rate (%)"):
    """
    绘制多组数据的折线图对比
    
    参数:
    data_dict: dict, 键为方法名称，值为数值列表
    title: str, 图表标题
    xlabel: str, x轴标签
    ylabel: str, y轴标签
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制每条折线
    for method, values in data_dict.items():
        # 将百分比字符串转换为浮点数
        # numeric_values = [float(v.strip('%')) for v in values]
        plt.plot(range(1, 1 + len(values)), values, marker='o', label=method)
    
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # 假设百分比范围是0-100%
    # plt.show()
    plt.savefig("/home/zby/Programs/Embodied_Analogy/scripts/draw_figs/paper_figs/explore_ablation.png")

# 示例数据
data = {
    "Ours Full": [
        25.00, 45.69, 56.90, 67.24, 73.28, 76.72, 77.59, 81.03, 81.90, 84.48,
        85.34, 85.34, 86.21, 86.21, 87.07, 87.07, 87.07, 87.07, 88.79, 88.79,
        88.79, 88.79, 88.79, 88.79, 89.66
    ],
    # "ours (pred)": [
    #     25.00, 45.69, 56.90, 68.97, 75.00, 78.45, 79.31, 84.48, 85.34, 87.93,
    #     88.79, 88.79, 89.66, 89.66, 90.52, 90.52, 90.52, 90.52, 92.24, 92.24,
    #     92.24, 92.24, 92.24, 92.24, 93.10
    # ],
    "Ours w\o IOR": [
        18.97, 34.48, 50.00, 56.90, 59.48, 61.21, 63.79, 64.66, 67.24, 68.97,
        69.83, 71.55, 72.41, 72.41, 73.28, 73.28, 74.14, 74.14, 75.00, 75.86,
        75.86, 76.72, 77.59, 77.59, 78.45
    ],
    # "ours w\o IOR (pred)": [
    #     19.83, 36.21, 51.72, 58.62, 62.07, 63.79, 67.24, 68.10, 70.69, 72.41,
    #     73.28, 75.00, 75.86, 75.86, 76.72, 76.72, 77.59, 77.59, 78.45, 79.31,
    #     79.31, 80.17, 81.03, 81.03, 81.90
    # ],
    "Ours w\o CA": [
        0.86, 7.76, 16.38, 37.07, 50.00, 54.31, 60.34, 62.93, 66.38, 68.97,
        75.86, 78.45, 82.76, 83.62, 83.62, 83.62, 85.34, 85.34, 87.93, 88.79,
        88.79, 88.79, 88.79, 88.79, 88.79
    ],
    # "ours w\o CA (pred)": [
    #     0.86, 8.62, 17.24, 39.66, 52.59, 56.90, 62.93, 65.52, 69.83, 72.41,
    #     79.31, 81.90, 86.21, 87.07, 87.07, 87.07, 88.79, 88.79, 91.38, 92.24,
    #     92.24, 92.24, 92.24, 92.24, 92.24
    # ]
    "Ours Zero-Shot": [
        23.28, 31.90, 45.69, 56.03, 56.90, 60.34, 63.79, 69.83, 71.55, 74.14,
        75.86, 76.72, 79.31, 80.17, 81.03, 82.76, 84.48, 84.48, 84.48, 85.34,
        85.34, 85.34, 86.21, 86.21, 86.21
    ]
}

# 调用函数绘制图表
plot_comparison(data, title="Ablation")