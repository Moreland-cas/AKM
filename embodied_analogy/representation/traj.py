# 专门用来存储探索过程中的 grasp 数据
import numpy as np
from embodied_analogy.utility.utils import dis_point_to_range

class Traj:
    def __init__(
        self,
        start_state=None,
        manip_type=None,
        pre_qpos=None, # NOTE pre_Tph2w 对应的 qpos
        Tph2w=None, # NOTE 并非实际 close gripper 后的 Tph2w, 而是由 grasp 转换来的 Tph2w
        goal_state=None,
        end_state=None,
        valid_mask=None
    ):
        # 初始化属性
        self.start_state = start_state    # 初始状态
        self.manip_type = manip_type      # 操作类型
        self.Tph2w = Tph2w                # 变换矩阵 (4, 4)
        self.pre_qpos = pre_qpos          # 抓取前的 qpos
        self.goal_state = goal_state      # 目标状态
        self.end_state = end_state        # 结束状态
        self.valid_mask = valid_mask      # 有效掩码
        
    def get_range(self):
        return (self.start_state, self.end_state)

    def __repr__(self):
        return (f"Traj(start_state={self.start_state}, "
                f"manip_type={self.manip_type}, "
                f"Tph2w={self.Tph2w}, "
                f"goal_state={self.goal_state}, "
                f"end_state={self.end_state}, "
                f"valid_mask={self.valid_mask})")

class Trajs:
    def __init__(self, joint_type):
        self.trajs = []
        
        assert joint_type in ["prismatic", "revolute"]
        self.joint_type = joint_type
        if joint_type == "prismatic":
            self.min_state = 0
            self.max_state = 0.5
        else:
            self.min_state = 0
            self.max_state = np.deg2rad(70)

    def merge_explored_ranges(self):
        """
        合并重叠或相邻的区间
        :param ranges: List[Tuple[float, float]]，输入的区间列表
        :return: List[Tuple[float, float]]，合并后的区间列表
        """
        if len(self.trajs) == 0:
            return []
        ranges = [traj.get_range() for traj in self.trajs]
        
        # 按区间起点排序
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [list(sorted_ranges[0])]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # 如果当前区间与最后一个合并区间有重叠或相邻
            if current_start <= last_end:
                merged[-1][1] = max(last_end, current_end)
            else:
                merged.append([current_start, current_end])
        
        # 转换回元组格式
        return [tuple(interval) for interval in merged]
    
    def get_ranges(self):
        """
        返回未被探索的区间
        :return: List[Tuple[float, float]]，未被探索的区间
        """
        merged = self.merge_explored_ranges()
        explored_ranges = merged
        unexplored_ranges = []
        
        # 定义目标区间的起始和结束
        target_start = self.min_state
        target_end = self.max_state
        
        # 如果没有任何已探索的区间，返回整个目标区间
        if not merged:
            return [(target_start, target_end)]
        
        # 检查目标区间的开始到第一个已探索区间的结束
        if target_start < merged[0][0]:
            unexplored_ranges.append((target_start, merged[0][0]))
        
        # 检查已探索区间之间的空隙
        for i in range(len(merged) - 1):
            unexplored_ranges.append((merged[i][1], merged[i + 1][0]))
        
        # 检查最后一个已探索区间的结束到目标区间的结束
        if merged[-1][1] < target_end:
            unexplored_ranges.append((merged[-1][1], target_end))
        
        return explored_ranges, unexplored_ranges
    
    def find_nearest_unexplored_range(self, cur_state):
        """
        找到距离 cur_state 最近的未探索区间, 且该区间需要足够的大, 太小的没必要探索了
        """
        # 首先获取所有的未探索区间
        _, unexplored_ranges = self.get_ranges()
        if len(unexplored_ranges) == 0:
            return None
        
        # 然后把特别小的未探索区间过滤掉
        filter_thresh = 0.05 if self.joint_type == "prismatic" else np.deg2rad(6)
        filtered_unexplored_ranges = []
        for unexplored_range in unexplored_ranges:
            if abs(unexplored_range[1] - unexplored_range[0]) > filter_thresh:
                filtered_unexplored_ranges.append(unexplored_range)
        if len(filtered_unexplored_ranges) == 0:
            return None
            
        # 最后找到距离当前状态最近的 range
        min_dist = 1e6
        min_range_idx = None
        for i, filtered_unexplored_range in enumerate(filtered_unexplored_ranges):
            cur_dist = dis_point_to_range(cur_state, filtered_unexplored_range)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_range_idx = i
        
        return filtered_unexplored_ranges[min_range_idx]
            
            
    def exist_large_unexplored_range(self):
        """
        判断当前是否还有比较大的 unexplored_range
        """
        if self.find_nearest_unexplored_range(0) is None:
            return False
        else:
            return True
        
    
    def is_range_covered(self):
        """
        判断区间列表的并集是否能覆盖目标区间
        :param ranges: List[Tuple[float, float]]，原始区间列表（不需要预合并）
        :param target: Tuple[float, float]，目标区间
        :return: True/False
        """
        merged = self.merge_explored_ranges()
        target_start, target_end = self.min_state, self.max_state
        current_coverage = target_start  # 当前已覆盖到的位置
        
        for start, end in merged:
            # 如果当前区间的起点已经超过需要覆盖的位置
            if start > current_coverage:
                return False
            
            # 如果当前区间能延长覆盖范围
            if end > current_coverage:
                current_coverage = end
            
            # 已经完全覆盖目标区间
            if current_coverage >= target_end:
                return True
        
        return current_coverage >= target_end
    
    def update(self, data):
        if isinstance(data, Traj):
            self.trajs.append(data)
        elif isinstance(data, list):
            for traj in data:
                self.trajs.append(traj)
    
    def is_current_state_explored(self, cur_state):
        """
        判断 cur_state 是否在已探索的区间中
        """
        merged = self.merge_ranges()
        for (start, end) in merged:
            if start <= cur_state <= end:
                return True
        return False
    
    def get_current_range(self, cur_state):
        """
            获取包含当前状态的区间, 有可能是 explored_range 或者是 unexplored_range
        """
        # 首先判断极端情况, 即 cur_state 小于 0 或者 大于 max_state
        if cur_state < self.min_state:
            return self.get_current_range(0)
        elif cur_state > self.max_state:
            return self.get_current_range(self.max_state)
        
        explored_ranges, unexplored_ranges = self.get_ranges()
        
        current_range = None
        is_explored = None
        if len(explored_ranges) != 0:
            for explore_range in explored_ranges:
                if explore_range[0] <= cur_state <= explore_range[1]:
                    current_range = explore_range
                    is_explored = True
                    break
        if len(unexplored_ranges) != 0:
            for unexplored_range in unexplored_ranges:
                if unexplored_range[0] <= cur_state <= unexplored_range[1]:
                    current_range = unexplored_range
                    is_explored = False
                    break
        if current_range is None:
            assert "Error in get_current_range(): cur_state not in any range whereas it should be"
        
        return is_explored, current_range 
        
    def check_neighbors(self, cur_state):
        # 这个函数一般只在 cur_state 处于 explore_range 中的时候调用
        """
        检查 cur_state 所在区间的左右紧挨的区间是否被探索过
        :param cur_state: 当前状态
        :return: (边界值, 方向字符串) 或 (None, None)
        """
        merged = self.merge_ranges()
        
        # 找到 cur_state 所在的区间
        current_range = None
        for (start, end) in merged:
            if start <= cur_state <= end:
                current_range = (start, end)
                break
        
        if current_range is None:
            assert "Error in check_neighbors(): cur_state not in any explored range whereas it should be"
        
        current_start, current_end = current_range
        left_explored = (current_start <= self.min_state)
        right_explored = (current_end >= self.max_state)
        
        # 根据情况返回结果
        if not left_explored and not right_explored:
            return current_start, "left", abs(current_start - self.min_state)
        elif not left_explored:
            return current_start, "left", abs(current_start - self.min_state)
        elif not right_explored:
            return current_end, "right", abs(current_end - self.max_state)
        return None, None, None
    
    def get_all_grasp(self):
        """
        返回一个列表 (state, grasp) 的列表
        """
        valid_grasp = []
        failed_grasp = []
        for traj in self.trajs:
            traj: Traj
            tmp_grasp = (traj.start_state, traj.pre_qpos)
            if traj.valid_mask == True:
                valid_grasp.append(tmp_grasp)
            else:
                failed_grasp.append(tmp_grasp)
        return valid_grasp, failed_grasp

    
    def __repr__(self):
        explrored_ranges, unexplrored_ranges = self.get_ranges()
        return f"Trajs(\
            \n\tmin_state={self.min_state}, max_state={self.max_state},\
            \n\texplored_ranges={explrored_ranges},\
            \n\tunexplrored_ranges={unexplrored_ranges},\
        )"
    
    
if __name__ == "__main__":
    pass
