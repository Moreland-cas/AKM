import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + ['DejaVu Serif', 'Times', 'serif']  # Fallback fonts
plt.rcParams.update({
    'font.size': 18,          # 全局默认字体大小
    # 'axes.titlesize': 16,     # 标题字体大小
    # 'axes.labelsize': 14,     # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,    # x轴刻度字体大小
    # 'ytick.labelsize': 12,    # y轴刻度字体大小
    # 'legend.fontsize': 12     # 图例字体大小
})



def plot_timeseries(data, xlabel="Number of Interactions (times)", ylabel="Exploration Success Rate (%)", output_file="/home/zby/Programs/AKM/scripts/draw_figs/paper_figs/explore_ablation_2.png"):
    """
    Plot time series with mean as lines and standard deviation as shaded error bands, and save as PNG.
    
    Parameters:
    - data: Dict with method names as keys and {'means': [...], 'stds': [...]} as values.
    - xlabel: Label for x-axis (default: 'Number of Interactions (times)').
    - ylabel: Label for y-axis (default: 'Exploration Success Rate (%)').
    - output_file: Path to save the PNG file (default: '/home/zby/Programs/AKM/scripts/draw_figs/paper_figs/explore_ablation_2.png').
    """
    # 创建图形
    plt.figure(figsize=(10, 5))
    
    # 高级感颜色列表（Tableau-inspired）
    colors = ['#4C78A8', '#F28C38', '#76B7B2', '#E15759', '#B07AA1', '#59A14F', '#EDC949']
    
    # 为每个方法绘制折线图和误差带
    for i, method in enumerate(data.keys()):
        means = data[method]['means']
        stds = data[method]['stds']
        time_stamps = range(1, 1 + len(means))
        
        # 绘制均值折线（无数据点）
        plt.plot(time_stamps, means, linestyle='-', color=colors[i % len(colors)], label=method)
        
        # 绘制标准差误差带（阴影区域）
        plt.fill_between(time_stamps, 
                         np.array(means) - np.array(stds), 
                         np.array(means) + np.array(stds), 
                         color=colors[i % len(colors)], alpha=0.2)
    
    # 设置轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.ylim(0, 100)  # 百分比范围 0-100%
    
    # 显示网格
    plt.grid(True, alpha=0.3)
    
    # 保存为 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # 关闭图形以释放内存
    plt.close()
    
    
# 示例用法
if __name__ == "__main__":
    
    method_list = [
        "ours_",
        "ours_wo_IOR_",
        "ours_wo_CA_",
        "ours_zs_",
    ]
    # 示例数据
    draw_data = {}
    result_path = "/home/zby/Programs/AKM/assets/analysis_batch"
    import sys
    sys.path.append("/home/zby/Programs/AKM/scripts/test_whole_pipeline/")
    from get_statistics import process_nested_dict
    for method in method_list:
        print("Method:", method)
        print()
        npy_list = []
        for idx in range(1, 6):
            data = np.load(os.path.join(result_path, f'{method}{idx}.npy'), allow_pickle=True).item()
            npy_list.append(data)
        print(process_nested_dict(npy_list))
        draw_data[method] = {}
        draw_data[method]['means'] = [process_nested_dict(npy_list)["Explore"][f"Tries_{i}"]["SR_5_5"]['mean'] for i in range(1, 26)]
        draw_data[method]['stds'] = [process_nested_dict(npy_list)["Explore"][f"Tries_{i}"]["SR_5_5"]['std'] for i in range(1, 26)]
    
    draw_data_new = {}
    draw_data_new["Ours (Full)"] = draw_data["ours_"]
    draw_data_new["Ours Zero-Shot"] = draw_data["ours_zs_"]
    draw_data_new["Ours w\o CA"] = draw_data["ours_wo_CA_"]
    draw_data_new["Ours w\o FS"] = draw_data["ours_wo_IOR_"]
    
    
    # 调用函数
    plot_timeseries(data=draw_data_new)
    pass
    