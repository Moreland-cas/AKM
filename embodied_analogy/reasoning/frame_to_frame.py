"""
    给定两帧铰链物体点云，和初始的 joint estimation, 输出更准的 joint estimation
    
    输入:
        depth1, depth2
        obj_mask1, obj_mask2
        joint_param, joint_state        
    
    迭代过程：
        对于 obj_mask1 中的像素进行分类
            假设 static part 和 moving part 对应的变换分别是 T_static 和 T_moving
            对于每个像素，计算其在变换后的三维位置 p_3d 和像素投影 p_2d
            
            正确的变换 => (distance(p_3d, camera_o) >= depth2[p_2d]) and (p_2d in obj_mask2)
            
        对于 obj_mask2 中的像素进行分类
            xxx
            
        找到 moving_mask1 和 moving_mask2 的交集 
            将 moving_mask1 中的像素进行 T_moving 变换, 并得到投影 moving_mask1_project_on2
            求 moving_mask1_project_on2 和 moving_mask2 的交集
            
        用交集的点进行 ICP 估计出 joint_param 和 joint_state
    
    输出:
        static_mask1, static_mask2
        moving_mask1, moving_mask2
        unknown_mask1, unknown_mask2
        joint_param, joint_state
"""