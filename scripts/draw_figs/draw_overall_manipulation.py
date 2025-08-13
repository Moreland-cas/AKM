import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 18,          # 全局默认字体大小
    # 'axes.titlesize': 16,     # 标题字体大小
    # 'axes.labelsize': 14,     # 坐标轴标签字体大小
    'xtick.labelsize': 12,    # x轴刻度字体大小
    'ytick.labelsize': 12,    # y轴刻度字体大小
    # 'legend.fontsize': 12     # 图例字体大小
})

def draw_colored_grid(data_dict, save_path, is_revolute=False):
    # 初始化一个4x4的矩阵，用于存储成功率
    matrix = np.zeros((4, 4))

    # 填充矩阵
    for key, value in data_dict.items():
        i, j = map(int, key.split("_"))
        matrix[3-j, i] = value
    
    normal_factor = 56 if is_revolute else 60
    matrix = matrix / normal_factor
    
    # 创建一个4x4的图
    plt.figure(figsize=(8, 8))
    sns.heatmap(matrix * 100, vmin=0, vmax=100, annot=True, fmt=".1f", cmap="Blues", cbar=False, linewidths=0.5)

    # 设置横轴和纵轴的标签
    plt.xticks(np.arange(4) + 0.5, [f"R{i}" for i in range(4)])
    plt.yticks(np.arange(4) + 0.5, [f"R{3-i}" for i in range(4)])

    plt.savefig(save_path)
    

if __name__ == "__main__":
    our_revolute = {
        "0_1": 46, "0_2": 45, "0_3": 41,
        "1_0": 41, "1_2": 36, "1_3": 37,
        "2_0": 41, "2_1": 31, "2_3": 35,
        "3_0": 41, "3_1": 37, "3_2": 27
    }
    our_prismatic = {
        "0_1": 44, "0_2": 45, "0_3": 44,
        "1_0": 44, "1_2": 41, "1_3": 39,
        "2_0": 45, "2_1": 41, "2_3": 34,
        "3_0": 38, "3_1": 41, "3_2": 36
    }
    draw_colored_grid(
        data_dict=our_revolute,
        save_path="/home/zby/Programs/AKM/paper_figs/our_revolute.png"
    )
    draw_colored_grid(
        data_dict=our_prismatic,
        save_path="/home/zby/Programs/AKM/paper_figs/our_prismatic.png"
    )
    
    gpnet_revolute = {
        "0_1": 6, "0_2": 5, "0_3": 5,
        "1_0": 8, "1_2": 5, "1_3": 9,
        "2_0": 11, "2_1": 7, "2_3": 8,
        "3_0": 7, "3_1": 6, "3_2": 8
    }
    gpnet_prismatic = {
        "0_1": 29, "0_2": 25, "0_3": 24,
        "1_0": 23, "1_2": 26, "1_3": 18,
        "2_0": 29, "2_1": 27, "2_3": 26,
        "3_0": 30, "3_1": 23, "3_2": 21
    }
    draw_colored_grid(
        data_dict=gpnet_revolute,
        save_path="/home/zby/Programs/AKM/paper_figs/gpnet_revolute.png"
    )
    draw_colored_grid(
        data_dict=gpnet_prismatic,
        save_path="/home/zby/Programs/AKM/paper_figs/gpnet_prismatic.png"
    )
    
    gflow_revolute = {
        "0_1": 6, "0_2": 6, "0_3": 3,
        "1_0": 7, "1_2": 2, "1_3": 3,
        "2_0": 7, "2_1": 2, "2_3": 5,
        "3_0": 8, "3_1": 4, "3_2": 5
    }
    gflow_prismatic = {
        "0_1": 23, "0_2": 15, "0_3": 15,
        "1_0": 32, "1_2": 24, "1_3": 18,
        "2_0": 27, "2_1": 28, "2_3": 22,
        "3_0": 25, "3_1": 21, "3_2": 19
    }
    draw_colored_grid(
        data_dict=gflow_revolute,
        save_path="/home/zby/Programs/AKM/paper_figs/gflow_revolute.png"
    )
    draw_colored_grid(
        data_dict=gflow_prismatic,
        save_path="/home/zby/Programs/AKM/paper_figs/gflow_prismatic.png"
    )