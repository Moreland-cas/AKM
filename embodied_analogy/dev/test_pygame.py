import pygame
import numpy as np

# 初始化pygame
pygame.init()

# 创建一个窗口
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Keyboard Control")

# 初始的delta值
delta = np.array([0, 0, 0])

running = True
while running:
    # 处理pygame的事件（例如窗口关闭）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取当前按键状态
    keys = pygame.key.get_pressed()
    delta = np.array([0, 0, 0])
    
    # 检查每个方向键，并更新delta
    if keys[pygame.K_UP]:
        delta = np.array([1, 0, 0])  # 向上（增加x轴）
    if keys[pygame.K_DOWN]:
        delta = np.array([-1, 0, 0])  # 向下（减少x轴）
    if keys[pygame.K_LEFT]:
        delta = np.array([0, 1, 0])   # 向左（增加y轴）
    if keys[pygame.K_RIGHT]:
        delta = np.array([0, -1, 0])  # 向右（减少y轴）
    if keys[pygame.K_KP_PLUS]:
        delta = np.array([0, 0, 1])   # 向前（增加z轴）
    if keys[pygame.K_KP_MINUS]:
        delta = np.array([0, 0, -1])  # 向后（减少z轴）
    
    # 打印当前delta值
    print(f"Current delta: {delta}")

    # 刷新显示窗口（防止界面冻结）
    screen.fill((0, 0, 0))  # 填充背景
    pygame.display.flip()    # 更新显示

    # 控制帧率
    pygame.time.Clock().tick(60)

# 退出pygame
pygame.quit()
