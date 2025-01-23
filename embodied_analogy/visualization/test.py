"""
Tracks 3D
=========

.. tags:: visualization-advanced
"""

import numpy as np
import napari


vertices = np.random.random(size=(10, 3)) * 10
colors = np.random.random(size=(10, 3))

def get_temporal_pc(t=100, num_points=10):
    data = np.zeros((num_points * t, 1 + 3))
    cur_pos = np.random.random((num_points, 3)) * 10
    for i in range(t):
        d_pos = np.random.random((num_points, 3))
        cur_pos += d_pos
        data[i * num_points: (i + 1) * num_points, 1:] = cur_pos
        data[i * num_points: (i + 1) * num_points, 0] = i
    return data
    
    
# data = get_temporal_pc()
viewer = napari.Viewer(ndisplay=3)
# viewer.add_points(data, size=1, name='points', opacity=0.3)
viewer.add_points(vertices, size=1, name='points', opacity=0.3, border_color=colors, face_color=colors)
# viewer.add_labels(labels, name='labels')
# viewer.add_tracks(tracks, features=features, name='tracks')

if __name__ == '__main__':
    napari.run()
