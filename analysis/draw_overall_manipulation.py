import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 输入的字典
data_dict = {
    "0_1": 0.3, "0_2": 0.7, "0_3": 0.1,
    "1_0": 0.4, "1_2": 0.6, "1_3": 0.2,
    "2_0": 0.5, "2_1": 0.8, "2_3": 0.9,
    "3_0": 0.6, "3_1": 0.7, "3_2": 0.4
}

# 初始化一个4x4的矩阵，用于存储成功率
matrix = np.zeros((4, 4))

# 填充矩阵
for key, value in data_dict.items():
    i, j = map(int, key.split("_"))
    matrix[i, j] = value

# 创建一个4x4的图
plt.figure(figsize=(8, 8))
sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False, linewidths=0.5)

# 设置横轴和纵轴的标签
plt.xticks(np.arange(4) + 0.5, [f"R{i}" for i in range(4)])
plt.yticks(np.arange(4) + 0.5, [f"R{i}" for i in range(4)])

# 设置空白
for i in range(4):
    plt.text(i + 0.5, i + 0.5, "-", ha='center', va='center', fontsize=12)

# 调整坐标轴方向
plt.gca().invert_yaxis()

# 显示图形
plt.show()
plt.savefig("/home/zby/Programs/Embodied_Analogy/paper_figs/overall_manipulation.png")