from skimage import data
import numpy as np
import napari

# create the viewer with an image
image = data.astronaut()[None]
image = np.tile(image, (10, 1, 1, 1)) # 10, 512, 512, 3
viewer = napari.view_image(image, rgb=True)
labels = np.random.randint(0, 3, size=(10, 512, 512))
viewer.add_labels(labels, name='labels')
if __name__ == '__main__':
    napari.run()