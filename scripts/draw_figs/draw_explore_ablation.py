import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/Programs/AKM/scripts/test_whole_pipeline/")
from get_statistics import process_nested_dict


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + ['DejaVu Serif', 'Times', 'serif']  # Fallback fonts
plt.rcParams.update({
    'font.size': 18, # Global default font size
    # 'axes.titlesize': 16, # Title font size
    # 'axes.labelsize': 14, # Axis label font size
    # 'xtick.labelsize': 12, # x-axis tick font size
    # 'ytick.labelsize': 12, # y-axis tick font size
    # 'legend.fontsize': 12 # Legend font size
})


def plot_timeseries(data, xlabel="Number of Interactions (times)", ylabel="Exploration Success Rate (%)", output_file="/home/Programs/AKM/scripts/draw_figs/paper_figs/explore_ablation_2.png"):
    """
    Plot time series with mean as lines and standard deviation as shaded error bands, and save as PNG.
    
    Parameters:
    - data: Dict with method names as keys and {'means': [...], 'stds': [...]} as values.
    - xlabel: Label for x-axis (default: 'Number of Interactions (times)').
    - ylabel: Label for y-axis (default: 'Exploration Success Rate (%)').
    - output_file: Path to save the PNG file (default: '/home/Programs/AKM/scripts/draw_figs/paper_figs/explore_ablation_2.png').
    """
    plt.figure(figsize=(10, 5))
    colors = ['#4C78A8', '#F28C38', '#76B7B2', '#E15759', '#B07AA1', '#59A14F', '#EDC949']
    
    # Draw line graphs and error bands for each method
    for i, method in enumerate(data.keys()):
        means = data[method]['means']
        stds = data[method]['stds']
        time_stamps = range(1, 1 + len(means))
        
        # Draw the mean line (no data points)
        plt.plot(time_stamps, means, linestyle='-', color=colors[i % len(colors)], label=method)
        
        # Draw standard deviation error bands (shaded areas)
        plt.fill_between(
            time_stamps, 
            np.array(means) - np.array(stds), 
            np.array(means) + np.array(stds), 
            color=colors[i % len(colors)], alpha=0.2
        )
    
    # set axis label
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.ylim(0, 100)  
    
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    
if __name__ == "__main__":
    method_list = [
        "ours_",
        "ours_wo_IOR_",
        "ours_wo_CA_",
        "ours_zs_",
    ]
    draw_data = {}
    result_path = "/home/Programs/AKM/assets/analysis_batch"
    
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
    
    plot_timeseries(data=draw_data_new)