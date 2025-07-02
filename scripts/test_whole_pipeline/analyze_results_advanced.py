"""
改为另外一种计算方式, 
用一个类存储一个 exp (task_setting, 具体到 i_j 的地步)
"""
import os
import json
import copy
import yaml
import sys
import numpy as np

################# HYPER_PARAMETERS #################
EXPLORE_PRISMATIC_VALID = 0.05 # m
EXPLORE_REVOLUTE_VALID = 5 # degree

PIVOT_THRESHS = [0.01, 0.05, 0.1, 0.2] # m
ANGLE_THRESHS = [1, 5, 10, 20] # degree

MANIP_RELATIVE_VALID_THRESHS = [0.1, 0.2, 0.5]
####################################################

class ExpDataPiece():
    def __init__(
        self,
        cfgs_path: str,         # "/home/zby/Programs/Embodied_Analogy/cfgs"
        run_name: str,          # 6_22
        obj_idx: str,           # 104
        range_transition: str   # i_j
    ):
        """
        根据 obj_folder_path 读取 explore_result 等
        """
        self.run_name = run_name
        self.obj_idx = obj_idx
        self.range_transition = range_transition
        
        run_yaml_path = os.path.join(cfgs_path, f"base_{run_name}.yaml")
        with open(run_yaml_path, "r") as f:
            run_yaml = yaml.safe_load(f)
        
        log_folder = run_yaml["exp_cfg"]["exp_folder"]
        run_path = os.path.join(log_folder, str(obj_idx))
        with open(os.path.join(run_path, f"{obj_idx}.yaml"), "r") as f:
            self.task_yaml = yaml.safe_load(f)
        
        with open(os.path.join(run_path, "explore_result.json"), "r") as f:
            explore_result = json.load(f)
            self.explore_result = explore_result
        
        with open(os.path.join(run_path, "recon_result.json"), "r") as f:
            recon_result = json.load(f)
            self.recon_result = recon_result
        
        with open(os.path.join(run_path, "manip_result.json"), "r") as f:
            manip_result = json.load(f)
            self.manip_result = manip_result[range_transition]
            
        self.joint_type = self.task_yaml["obj_env_cfg"]["joint_type"]
    
    def explore_actually_valid(self):
        """
        输入的是一个 has_valid_explore 为 True 的 explore_result, 本函数判断该 result 是否是真的 valid
        """
        self.pred_explore_valid = self.explore_result["has_valid_explore"]
        if not self.pred_explore_valid:
            return False
        
        joint_delta = self.explore_result["joint_state_end"] - self.explore_result["joint_state_start"]
        # NOTE 这里没有给 joint_delta 加 abs, 因为我们就是要 open
        if self.joint_type == "prismatic":
            return joint_delta >= EXPLORE_PRISMATIC_VALID
        
        elif self.joint_type == "revolute":
            return np.rad2deg(joint_delta) >= EXPLORE_REVOLUTE_VALID
    
    def coarse_reconstruct_valid_under_thr(self, angle_thr_deg, pivot_thr_m):
        """
        判断 reconstruct 是否有效
        """
        if not self.actual_explore_valid:
            return False
        
        # reconstruct 阶段报错
        if not self.recon_result["has_valid_recon"]:
            return False
        
        if not self.reconstruct_type_right():
            return False
        
        angle_loss_c = self.recon_result['coarse_loss']['angle_err']
        pivot_loss_c = self.recon_result['coarse_loss']['pos_err']
        
        # 根据重建结果判断 recon_valid
        if self.joint_type == "prismatic":
            return angle_loss_c < np.deg2rad(angle_thr_deg)
        elif self.joint_type == "revolute":
            return pivot_loss_c < pivot_thr_m and angle_loss_c < np.deg2rad(angle_thr_deg)
        
    def fine_reconstruct_valid_under_thr(self, angle_thr_deg, pivot_thr_m):
        """
        判断 reconstruct 是否有效
        """
        if not self.actual_explore_valid:
            return False
        
        # reconstruct 阶段报错
        if not self.recon_result["has_valid_recon"]:
            return False
        
        if not self.reconstruct_type_right():
            return False
        
        angle_loss_f = self.recon_result['fine_loss']['angle_err']
        pivot_loss_f = self.recon_result['fine_loss']['pos_err']
        
        # 根据重建结果判断 recon_valid
        if self.joint_type == "prismatic":
            return angle_loss_f < np.deg2rad(angle_thr_deg)
        elif self.joint_type == "revolute":
            return pivot_loss_f < pivot_thr_m and angle_loss_f < np.deg2rad(angle_thr_deg)
    
    def reconstruct_type_right(self):
        if not self.actual_explore_valid:
            return False
        if not self.recon_result["has_valid_recon"]:
            return False
        fine_type_loss = self.recon_result['fine_loss']['type_err']
        return fine_type_loss == 0
        
    def manipulate_valid_under_thr(self, num_manip, rel_thr):
        """
        根据 relative error 判断 manipulate 是否成功
        num_manip: 经过 num_manip 次操作后, 是否成功
        """
        manip_distance = abs(self.manip_result["manip_end_state"] - self.manip_result["manip_start_state"])
        
        time_steps = self.manip_result.keys()
        time_steps = list(time_steps)
        # delete 两个字符串的 key
        try:
            time_steps.remove("manip_start_state")
            time_steps.remove("manip_end_state")
            time_steps.remove("exception")
        except Exception as e:
            pass
        
        time_steps = sorted(time_steps)
        self.loss_array = [abs(self.manip_result[time_step]["diff"]) for time_step in time_steps]
        self.last_loss = self.loss_array[-1]
        
        if num_manip > len(self.loss_array) - 1:
            cur_loss = self.last_loss
        else:
            cur_loss = self.loss_array[num_manip]
        relative_error = cur_loss / manip_distance
        return relative_error < rel_thr

    def process(self):
        """
        根据初始化得到的信息进行初步的处理, 得到一些统计量
        """
        if "num_tries" in self.explore_result:
            self.num_tries = self.explore_result["num_tries"]
        else:
            self.num_tries = self.task_yaml["explore_env_cfg"]["max_tries"]
            
        self.actual_explore_valid = self.explore_actually_valid()
        
        self.pred_type_right = self.reconstruct_type_right()
        self.coarse_recon_valid = [self.coarse_reconstruct_valid_under_thr(
            angle_thr_deg=angle_thr_deg,
            pivot_thr_m=pivot_thr_m
        ) for (angle_thr_deg, pivot_thr_m) in zip(ANGLE_THRESHS, PIVOT_THRESHS)]
        self.fine_recon_valid = [self.fine_reconstruct_valid_under_thr(
            angle_thr_deg=angle_thr_deg,
            pivot_thr_m=pivot_thr_m
        ) for (angle_thr_deg, pivot_thr_m) in zip(ANGLE_THRESHS, PIVOT_THRESHS)]
        
        self.manip_valid = []
        for i in range(self.task_yaml["manip_env_cfg"]["max_attempts"] + 1):
            self.manip_valid.append([
                self.manipulate_valid_under_thr(num_manip=i, rel_thr=thr) for thr in MANIP_RELATIVE_VALID_THRESHS
            ])
        
        # 将 self.loss_array padd 到 max_tries
        self.num_closed_loop_manip = len(self.loss_array)
        # NOTE 之所以要 +1, 是因为我们还保存了操作 0 次时的 loss
        for i in range(self.task_yaml["manip_env_cfg"]["max_attempts"] - self.num_closed_loop_manip + 1):
            self.loss_array.append(self.last_loss)
        
        # 记录一个 长度 为 max_tries 的 manip_error list
        # TODO reconstruct 还得记录一下 coarse 和 fine 的 angle/pivot loss
        # 对于 reconstruct 只对于对的统计定量的平均值
        # TODO 还得有个不同 closed-loop 时候的 manip_loss
        
def print_frac(prefix, numerator, denominator, use_percentage=True):
    frac = numerator / denominator
    frac_percentage = frac * 100
    if use_percentage:
        print(f"{prefix}{numerator} / {denominator} = {frac_percentage:.2f}%")
    else:
        print(f"{prefix}{numerator} / {denominator} = {frac:.2f}")
        
class DataAnalyze():
    def __init__(self, datalist):
        self.datalist = datalist
        for datapiece in self.datalist:
            datapiece.process()
        self.num_total_exp = len(self.datalist)
        self.num_prismatic_exp = sum([int(datapiece.joint_type == "prismatic") for datapiece in self.datalist])
        self.num_revolute_exp = sum([int(datapiece.joint_type == "revolute") for datapiece in self.datalist])
    
    def summary_explore(self):
        """
        pred_sr
        actual_sr
        num_tries
        """
        max_tries = self.datalist[0].task_yaml["explore_env_cfg"]["max_tries"]
        tries_sum = 0
        
        # 全局成功率统计
        num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
        num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}
        
        # 处理每个任务的结果
        for datapiece in self.datalist:
            datapiece: ExpDataPiece
            tries_sum += datapiece.num_tries
    
            for i in range(datapiece.num_tries, max_tries + 1):
                num_pred_valid[i] += datapiece.pred_explore_valid
                num_actual_valid[i] += datapiece.actual_explore_valid
            
        print("\n************** Explore Stage Analysis **************")
        print_frac("SR (actual): ", num_actual_valid[max_tries], self.num_total_exp)
        print_frac("SR (pred): ", num_pred_valid[max_tries], self.num_total_exp)
        print_frac("Num Tries (all): ", tries_sum, self.num_total_exp, use_percentage=False)
    
        print("\nDetailed Results (All Joints):")
        for tries in range(1, max_tries + 1):
            print(f"tries={tries}:")
            print_frac("\tSR (actual): ", num_actual_valid[tries], self.num_total_exp)
            print_frac("\tSR (pred): ", num_pred_valid[tries], self.num_total_exp)
        print("****************************************************\n")

    def summary_recon(self):
        """
        统计:
        type_pred_sr
        在不同 thresh 下的 recon_sr (分为 coarse 和 fine 统计)
        """
        print("************** Reconstruct Stage Analysis **************")
        datalist_filtered = self.datalist
        sum_pred_type_right = sum([int(datapiece.pred_type_right) for datapiece in datalist_filtered])
        print_frac("Joint type pred SR: ", sum_pred_type_right, self.num_total_exp)
        
        for i, (angle_thr, pivot_thr) in enumerate(zip(ANGLE_THRESHS, PIVOT_THRESHS)):
            print(f"Angle error threshold: {angle_thr} degree, Pivot error threshold: {pivot_thr} m")
            print_frac("\tSR (coarse): ", sum([int(datapiece.coarse_recon_valid[i]) for datapiece in datalist_filtered]), self.num_total_exp)
            print_frac("\tSR (fine): ", sum([int(datapiece.fine_recon_valid[i]) for datapiece in datalist_filtered]), self.num_total_exp)
        print("****************************************************")
        
    
    def summary_manip_helper(self, range_transition_delta=None):
        """
        对于所有的, 和分别对于每个 Ri to Rj
        统计:
        所有 exp 的平均 manip_error (包含 close_loop)
        不同 thresh 下的: manip_sr
        """
        if range_transition_delta is None:
            datalist_filtered = self.datalist
        else:
            datalist_filtered = []
            for datapiece in self.datalist:
                i, j = map(int, datapiece.range_transition.split("_"))
                if abs(i-j) == range_transition_delta:
                    datalist_filtered.append(datapiece)
            
        # 打印平均误差, 分为 revolute 和 prismatic 打印
        revolute_loss_array = []
        prismatic_loss_array = []
        for datapiece in datalist_filtered:
            if datapiece.joint_type == "prismatic":
                prismatic_loss_array.append(datapiece.loss_array)
            else:
                revolute_loss_array.append(datapiece.loss_array)
        
        if range_transition_delta is None: 
            print("Summary for Total:")
        else:
            print(f"Summary for {range_transition_delta}: ")
        
        print("Revolute  closed-loop loss: ")
        revolute_loss_list = [f"{np.rad2deg(loss):.2f}deg" for loss in np.array(revolute_loss_array).mean(0)]
        print(f"\t{' -> '.join(revolute_loss_list)}")
        
        print("Prismatic closed-loop loss: ")
        prismatic_loss_list = [f"{100 * loss:.2f}cm" for loss in np.array(prismatic_loss_array).mean(0)]
        print(f"\t{' -> '.join(prismatic_loss_list)}")
        
        # 打印成功功率
        for i, manip_thr in enumerate(MANIP_RELATIVE_VALID_THRESHS):
            # 对于一个 thresh 进行打印
            closed_loop_sr = []
            for j in range(6):
                tmp_sr = sum([int(datapiece.manip_valid[j][i]) for datapiece in datalist_filtered]) / len(datalist_filtered)
                closed_loop_sr.append(f"{tmp_sr * 100:.2f}%")
            print(f"Manip relative threshold: {manip_thr}")
            print(f"\t{' -> '.join(closed_loop_sr)}")
    
    def summary_manip(self):
        print("\n************** Manipulation Stage Analysis **************")
        self.summary_manip_helper()
        # self.summary_manip_helper(range_transition_delta=1)
        # self.summary_manip_helper(range_transition_delta=2)
        # self.summary_manip_helper(range_transition_delta=3)
        print("****************************************************")
        
    
def analyze(run_name="6_21"):
    data_list = []
    for obj_idx in range(116):
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                data_list.append(
                    ExpDataPiece(
                        cfgs_path="/home/zby/Programs/Embodied_Analogy/cfgs",
                        run_name=run_name,          
                        obj_idx=obj_idx,         
                        range_transition=f"{i}_{j}"
                    )
                )
    dataAnalyze = DataAnalyze(datalist=data_list)
    save_analysis_path = f"/home/zby/Programs/Embodied_Analogy/assets/analysis_advanced/{run_name}.txt"
    with open(save_analysis_path, "w") as f:
        sys.stdout = f
        dataAnalyze.summary_explore()
        dataAnalyze.summary_recon()
        dataAnalyze.summary_manip()

if __name__ == "__main__":
    run_names = [
        # "6_18",
        # "6_21",
        # "6_26",
        # "6_27",
        # "6_29",
        # "6_30",
        # "6_31",
        # "6_32",
        # "6_33",
        "6_34"
    ]
    for run_name in run_names:
        analyze(run_name)
        