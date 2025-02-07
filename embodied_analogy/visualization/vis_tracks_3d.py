"""
    可视化 3D 的 tracks
"""
import torch
import numpy as np
import napari

def vis_tracks3d_napari(tracks_3d, colors=None):
    """
    Args:
        tracks_3d: np.array of shape (T, M, 3)
        colors: np.array of shape (M, 3)
    """
    # 首先将输入的 tracks_3d 转换为 napari 支持的格式, 即 T * M, (1t+3d) 的形式
    if isinstance(tracks_3d, torch.Tensor):
        tracks_3d = tracks_3d.cpu().numpy()
        
    T, M, _ = tracks_3d.shape
    napari_data = np.zeros((T * M, 1 + 3))
    for i in range(T):
        napari_data[i * M: (i + 1) * M, 0] = i
        napari_data[i * M: (i + 1) * M, 1:] = tracks_3d[i]
    
    viewer = napari.Viewer(ndisplay=3)
    
    if colors is None:
        colors = np.random.rand(M, 3)
        # 将 M, 3 大小的 colors 变换为 T*M, 3 的大小
    colors = np.tile(colors, (T, 1))
        
    viewer.add_points(napari_data, size=0.02, name='tracks_3d', opacity=0.8, face_color=colors)
    napari.run()
    
def vis_pointcloud_series_napari(pc_series, colors=None):
    """
    Args:
        pc_series: list of np.arrays of shape (M, 3)
        colors: list of np.arrays of shape (M, 3)
    """
    # 首先将输入的 pc_series 转换为 napari 支持的格式, 即 T * M, (1t+3d) 的形式
    # if isinstance(pc_series, torch.Tensor):
    #     pc_series = pc_series.cpu().numpy()
        
    T = len(pc_series)
    napari_data = []
    for i in range(T):
        pc = pc_series[i] # M, 3
        pc_with_t = np.concatenate([np.ones((pc.shape[0], 1)) * i, pc], axis=1) # M, 4
        napari_data.append(pc_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    
    viewer = napari.Viewer(ndisplay=3)
    # 将 M, 3 大小的 colors 变换为 T*M, 3 的大小
    if colors is not None:
        colors = np.concatenate(colors, axis=0)
        viewer.add_points(napari_data, size=0.02, name='pc_series', opacity=0.8, face_color=colors)
    else:
        viewer.add_points(napari_data, size=0.02, name='pc_series', opacity=0.8)
    napari.run()

if __name__ == '__main__':
    pc_series = np.random.random((100, 10, 3)) * 10
    vis_pointcloud_series_napari(pc_series)