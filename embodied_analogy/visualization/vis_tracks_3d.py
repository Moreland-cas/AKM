"""
    可视化 3D 的 tracks
"""
import torch
import numpy as np
import napari

def vis_tracks_3d_napari(pc_series, colors):
    """
    Args:
        pc_series: np.array of shape (T, M, 3)
        colors: np.array of shape (M, 3)
    """
    # 首先将输入的 pc_series 转换为 napari 支持的格式, 即 T * M, (1t+3d) 的形式
    if isinstance(pc_series, torch.Tensor):
        pc_series = pc_series.cpu().numpy()
        
    T, M, _ = pc_series.shape
    napari_data = np.zeros((T * M, 1 + 3))
    for i in range(T):
        napari_data[i * M: (i + 1) * M, 0] = i
        napari_data[i * M: (i + 1) * M, 1:] = pc_series[i]
    
    viewer = napari.Viewer(ndisplay=3)
    # 将 M, 3 大小的 colors 变换为 T*M, 3 的大小
    colors = np.tile(colors, (T, 1))
    viewer.add_points(napari_data, size=0.02, name='tracks_3d', opacity=0.8, face_color=colors)
    # viewer.add_labels(np.random.randint(0, 10, size=(M, 1)), name='labels')
    napari.run()

if __name__ == '__main__':
    pc_series = np.random.random((100, 10, 3)) * 10
    vis_tracks_3d_napari(pc_series)