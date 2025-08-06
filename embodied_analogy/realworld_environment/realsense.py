import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置深度流 (640x480, 30 FPS) 和 RGB 流 (640x480, 30 FPS)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
profile = pipeline.start(config)

"""
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print('%02d: %s'%(i,visulpreset))
---------------------------------------
00: Custom
01: Default
02: Hand
03: High Accuracy
04: High Density
"""
# 设置深度预设 (Default)
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 1)     # Default
depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

# 创建滤波器
dec_filter = rs.decimation_filter()  # 解码滤波
dec_filter.set_option(rs.option.filter_magnitude, 2)  

spatial_filter = rs.spatial_filter()  # 空间滤波
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
spatial_filter.set_option(rs.option.holes_fill, 0)

temporal_filter = rs.temporal_filter()  # 时间滤波
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
"""
preset_range = temporal_filter.get_option_range(rs.option.holes_fill)
for i in range(int(preset_range.max)+1):
    visulpreset = temporal_filter.get_option_value_description(rs.option.holes_fill, i)
    print('%02d: %s'%(i,visulpreset))
-----------------------------------------------------
00: Disabled
01: Valid in 8/8
02: Valid in 2/last 3
03: Valid in 2/last 4
04: Valid in 2/8
05: Valid in 1/last 2
06: Valid in 1/last 5
07: Valid in 1/8
08: Always on
"""
temporal_filter.set_option(rs.option.holes_fill, 7)  # Valid in 1/8

threshold_filter = rs.threshold_filter()  # 阈值滤波
threshold_filter.set_option(rs.option.min_distance, 0.2)  # 最小深度 0.1 米
threshold_filter.set_option(rs.option.max_distance, 4.0)

hole_filling = rs.hole_filling_filter()  # 孔洞填充
"""
00: Fill from Left
01: Farest from around
02: Nearest from around
"""
hole_filling.set_option(rs.option.holes_fill, 1)

# 创建对齐对象
align_to_color = rs.align(rs.stream.color)

# 获取 RGB 传感器并启用自动曝光
# rgb_sensor = None
# for sensor in profile.get_device().query_sensors():
#     if sensor.get_info(rs.camera_info.name) == "RGB Camera":
#         rgb_sensor = sensor
#         break
# if rgb_sensor:
    
rgb_sensor = profile.get_device().first_color_sensor()
rgb_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 应用滤波
        # depth_frame = dec_filter.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = threshold_filter.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        
        # 深度到 RGB 对齐
        # aligned_frames = align_to_color.process(frames)
        # aligned_depth_frame = aligned_frames.get_depth_frame()
        # color_frame = aligned_frames.get_color_frame()

        # 应用孔洞填充
        # aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        # 转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度图归一化显示
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
        )

        # 显示图像
        cv2.imshow("RGB Image", color_image)
        cv2.imshow("Aligned Depth Image", depth_colormap)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()