def merge_ranges(ranges):
    """合并重叠区间"""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(sorted_ranges[0])]
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
    return [tuple(x) for x in merged]

def find_current_interval(state, merged_ranges):
    """找到当前状态所在的区间"""
    for start, end in merged_ranges:
        if start <= state <= end:
            return (start, end)
    return None

def smart_exploration(explored_ranges, target_range, state_min, state_max):
    """
    新版探索策略核心算法
    :param explored_ranges: 当前已探索的区间列表
    :param target_range: 目标覆盖区间 (target_min, target_max)
    :return: 更新后的explored_ranges
    """
    target_min, target_max = target_range
    merged = merge_ranges(explored_ranges)
    current_state = get_current_state()  # 获取当前实际状态
    
    # 判断当前状态是否在已探索范围内
    current_interval = find_current_interval(current_state, merged)
    
    if current_interval is not None:
        # ---------- 情况1：在已探索范围内 ----------
        start, end = current_interval
        
        # 移动到当前区间的边界（优先左边界）
        if abs(current_state - start) > 1e-6:
            execute_move(direction="left", step=current_state - start)
        
        # 获取更新后的状态
        current_state = get_current_state()
        
        # 计算需要探索的左右边界
        need_left = start > target_min
        need_right = end < target_max
        
        # 优先探索左边
        if need_left:
            # 向左拉满探索
            step = start - target_min
            actual = execute_move(direction="left", step=step)
            if actual < start - 1e-6:  # 有效移动
                explored_ranges.append((actual, start))
        
        # 再探索右边
        if need_right and end < target_max:
            step = target_max - end
            actual = execute_move(direction="right", step=step)
            if actual > end + 1e-6:  # 有效移动
                explored_ranges.append((end, actual))
                
    else:
        # ---------- 情况2：在已探索范围外 ----------
        # 优先向左探索
        step_left = current_state - target_min
        if step_left > 1e-6:
            actual_left = execute_move(direction="left", step=step_left)
            explored_ranges.append((actual_left, current_state))
        
        # 如果左边不可行则向右
        if not is_range_covered(explored_ranges, target_range):
            step_right = target_max - current_state
            actual_right = execute_move(direction="right", step=step_right)
            explored_ranges.append((current_state, actual_right))
    
    # 最后合并区间并返回
    return merge_ranges(explored_ranges)

def is_range_covered(explored_ranges, target_range):
    """判断是否已覆盖目标区间"""
    merged = merge_ranges(explored_ranges)
    if not merged:
        return False
    return merged[0][0] <= target_range[0] and merged[-1][1] >= target_range[1]