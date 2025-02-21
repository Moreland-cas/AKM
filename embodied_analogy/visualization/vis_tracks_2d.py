"""
    用于可视化 tracks_2d, 给定一系列背景图片(如 rgb_frames 或者 seg_mask)
"""
import torch
import numpy as np


def vis_tracks2d_napari(image_frames, tracks_2d, colors=None, viewer_title="napari"):
    """
    Args:
        image_frames: np.array([T, H, W, C])
        tracks_2d: np.array of shape (T, M, 2), (u, v)
        colors: np.array of shape (M, 3)
    """
    if isinstance(tracks_2d, torch.Tensor):
        tracks_2d = tracks_2d.cpu().numpy()
        
    T, M, _ = tracks_2d.shape
    import napari
    viewer = napari.view_image(image_frames, rgb=True)
    viewer.title = viewer_title
    
    # 把 tracks_2d 转换成 napari 支持的格式
    napari_data = []
    for i in range(T):
        track_2d = tracks_2d[i] # M, 2
        track_2d_with_t = np.concatenate([np.ones((track_2d.shape[0], 1)) * i, track_2d], axis=1) # M, 3
        # 把 (u, v) 两个维度交换
        track_2d_with_t = track_2d_with_t[:, [0, 2, 1]]
        napari_data.append(track_2d_with_t)
    napari_data = np.concatenate(napari_data, axis=0)
    
    if colors is None:
        colors = np.random.rand(M, 3)
        # 将 M, 3 大小的 colors 变换为 T*M, 3 的大小
    colors = np.tile(colors, (T, 1))
        
    viewer.add_points(napari_data, size=1, name='tracks_2d', opacity=1., face_color=colors)
    
    napari.run()

def test_vis_tracks2d_napari():
    # 生成模拟图像帧数据 (5帧，100x100像素，RGB)
    T, H, W, C = 5, 100, 100, 3
    image_frames = np.random.randint(0, 255, (T, H, W, C), dtype=np.uint8)

    # 生成模拟2D轨迹数据 (5帧，10个点，每个点的(u, v))
    M = 10
    tracks_2d = np.random.randint(0, 100, (T, M, 2))

    # 生成颜色数据 (10个点，每个点的RGB颜色)
    colors = np.random.rand(M, 3)
    vis_tracks2d_napari(image_frames, tracks_2d)


if __name__ == "__main__":
    test_vis_tracks2d_napari()
