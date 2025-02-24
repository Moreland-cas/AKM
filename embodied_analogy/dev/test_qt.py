import ctypes
import sys
import os
import numpy as np

import cv2
import napari
import cv2
# cv2.waitKey(0)
viewer = napari.Viewer()
viewer.add_image(np.random.random((100, 100)))
napari.run()

from PyQt5.QtCore import QCoreApplication, QLibraryInfo

# 在代码开头添加以下配置
os.environ["QT_DEBUG_PLUGINS"] = "1"  # 开启插件调试
os.environ["QT_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
print(os.environ["QT_PLUGIN_PATH"])