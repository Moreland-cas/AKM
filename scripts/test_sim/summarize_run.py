import os
import json
import yaml
import sys
import argparse
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
        cfgs_path: str,         # "/home/Programs/AKM/cfgs"
        run_name: str,          # 6_22
        obj_idx: str,           # 104
        range_transition: str   # i_j
    ):
        self.run_name = run_name
        self.obj_idx = obj_idx
        self.range_transition = range_transition
        
        run_yaml_path = os.path.join(cfgs_path, f"{run_name}.yaml")
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
        The input is an explore_result with has_valid_explore set to True. This function checks whether the result is valid.
        """
        self.pred_explore_valid = self.explore_result["has_valid_explore"]
        if not self.pred_explore_valid:
            return False
        
        joint_delta = self.explore_result["joint_state_end"] - self.explore_result["joint_state_start"]
        # NOTE We don't add abs to joint_delta here, because we just want to open
        if self.joint_type == "prismatic":
            return joint_delta >= EXPLORE_PRISMATIC_VALID
        
        elif self.joint_type == "revolute":
            return np.rad2deg(joint_delta) >= EXPLORE_REVOLUTE_VALID
    
    def coarse_reconstruct_valid_under_thr(self, angle_thr_deg, pivot_thr_m):
        if not self.actual_explore_valid:
            return False
        
        if not self.recon_result["has_valid_recon"]:
            return False
        
        if not self.reconstruct_type_right():
            return False
        
        angle_loss_c = self.recon_result['coarse_loss']['angle_err']
        pivot_loss_c = self.recon_result['coarse_loss']['pos_err']
        
        # Determine recon_valid based on the reconstruction result
        if self.joint_type == "prismatic":
            return angle_loss_c < np.deg2rad(angle_thr_deg)
        elif self.joint_type == "revolute":
            return pivot_loss_c < pivot_thr_m and angle_loss_c < np.deg2rad(angle_thr_deg)
        
    def fine_reconstruct_valid_under_thr(self, angle_thr_deg, pivot_thr_m):
        if not self.actual_explore_valid:
            return False
        
        if not self.recon_result["has_valid_recon"]:
            return False
        
        if not self.reconstruct_type_right():
            return False
        
        angle_loss_f = self.recon_result['fine_loss']['angle_err']
        pivot_loss_f = self.recon_result['fine_loss']['pos_err']
        
        # Determine recon_valid based on the reconstruction result
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
        Determines whether the manipulation was successful based on the relative error.
        num_manip: After num_manip operations, whether the manipulation was successful.
        """
        manip_distance = abs(self.manip_result["manip_end_state"] - self.manip_result["manip_start_state"])
        
        time_steps = self.manip_result.keys()
        time_steps = list(time_steps)
        # delete the key of two strings
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
        Perform preliminary processing based on the information obtained from initialization to obtain some statistics
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
        for i in range(self.task_yaml["manip_env_cfg"]["max_manip"] + 1):
            self.manip_valid.append([
                self.manipulate_valid_under_thr(num_manip=i, rel_thr=thr) for thr in MANIP_RELATIVE_VALID_THRESHS
            ])
        
        # padd self.loss_array to max_tries
        self.num_closed_loop_manip = len(self.loss_array)
        # NOTE The reason for +1 is that we also save the loss when the operation is 0 times
        for i in range(self.task_yaml["manip_env_cfg"]["max_manip"] - self.num_closed_loop_manip + 1):
            self.loss_array.append(self.last_loss)
        
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
        
        # for saving
        self.processed_data_dict = {
            "Explore": {
                "SR_pred": 0,
                "SR_5_5": 0,
                "num_tries": 0,
                "Tries_1": {
                    "SR_pred": 0,
                    "SR_5_5": 0,
                }
                # ...
            },
            "Modeling": {
                "Type_acc": 0,
                "coarse": { 
                    "SR_1_1": 0,
                    "SR_5_5": 0,
                    "SR_10_10": 0,
                    "SR_20_20": 0,
                },
                "fine": { 
                    "SR_1_1": 0,
                    "SR_5_5": 0,
                    "SR_10_10": 0,
                    "SR_20_20": 0,
                },
            },
            "Manipulation": {
                "0th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
                "1th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
                "2th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
                "3th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
                "4th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
                "5th": {
                    "SR_0.1": 0,
                    "SR_0.2": 0,
                    "SR_0.5": 0,
                },
            }
        }
    
    def summary_explore(self):
        """
        pred_sr
        actual_sr
        num_tries
        """
        max_tries = self.datalist[0].task_yaml["explore_env_cfg"]["max_tries"]
        tries_sum = 0
        
        # Global success rate statistics
        num_pred_valid = {i: 0 for i in range(1, max_tries + 1)}
        num_actual_valid = {i: 0 for i in range(1, max_tries + 1)}
        
        # Process the results of each task
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
            
            # save data
            if f'Tries_{tries}' not in self.processed_data_dict["Explore"]:
                self.processed_data_dict["Explore"][f'Tries_{tries}'] = {}
            self.processed_data_dict["Explore"][f'Tries_{tries}']["SR_pred"] = num_pred_valid[tries] / self.num_total_exp * 100
            self.processed_data_dict["Explore"][f'Tries_{tries}']["SR_5_5"] = num_actual_valid[tries] / self.num_total_exp * 100
        
        print("****************************************************\n")
        
        # save data
        self.processed_data_dict["Explore"]["SR_pred"] = num_pred_valid[max_tries] / self.num_total_exp * 100
        self.processed_data_dict["Explore"]["SR_5_5"] = num_actual_valid[max_tries] / self.num_total_exp * 100
        self.processed_data_dict["Explore"]["num_tries"] = tries_sum / self.num_total_exp

    def summary_recon(self):
        """
        Statistics:
        type_pred_sr
        recon_sr at different thresh levels (divided into coarse and fine statistics)
        """
        print("************** Reconstruct Stage Analysis **************")
        datalist_filtered = self.datalist
        sum_pred_type_right = sum([int(datapiece.pred_type_right) for datapiece in datalist_filtered])
        print_frac("Joint type pred SR: ", sum_pred_type_right, self.num_total_exp)
        
        # for data save
        self.processed_data_dict["Modeling"]["Type_acc"] = sum_pred_type_right / self.num_total_exp * 100
        
        for i, (angle_thr, pivot_thr) in enumerate(zip(ANGLE_THRESHS, PIVOT_THRESHS)):
            print(f"Angle error threshold: {angle_thr} degree, Pivot error threshold: {pivot_thr} m")
            print_frac("\tSR (coarse): ", sum([int(datapiece.coarse_recon_valid[i]) for datapiece in datalist_filtered]), self.num_total_exp)
            print_frac("\tSR (fine): ", sum([int(datapiece.fine_recon_valid[i]) for datapiece in datalist_filtered]), self.num_total_exp)
            
            # for data save
            self.processed_data_dict["Modeling"]["coarse"][f'SR_{int(angle_thr)}_{int(pivot_thr * 100)}'] = sum([int(datapiece.coarse_recon_valid[i]) for datapiece in datalist_filtered]) / self.num_total_exp * 100
            self.processed_data_dict["Modeling"]["fine"][f'SR_{int(angle_thr)}_{int(pivot_thr * 100)}'] = sum([int(datapiece.fine_recon_valid[i]) for datapiece in datalist_filtered]) / self.num_total_exp * 100
        print("****************************************************")
        
    def summary_manip_helper(self, range_transition_delta=None):
        """
        For all, and separately for each Ri to Rj
        Statistics:
        Average manip_error for all exp (including close_loop)
        Manip_sr for different thresh values:
        """
        if range_transition_delta is None:
            datalist_filtered = self.datalist
        else:
            datalist_filtered = []
            for datapiece in self.datalist:
                i, j = map(int, datapiece.range_transition.split("_"))
                if abs(i-j) == range_transition_delta:
                    datalist_filtered.append(datapiece)
            
        # Print average error, divided into revolute and prismatic printing
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
        
        # Print successful power
        for i, manip_thr in enumerate(MANIP_RELATIVE_VALID_THRESHS):
            closed_loop_sr = []
            for j in range(6):
                tmp_sr = sum([int(datapiece.manip_valid[j][i]) for datapiece in datalist_filtered]) / len(datalist_filtered)
                closed_loop_sr.append(f"{tmp_sr * 100:.2f}%")
                
                # for data save
                self.processed_data_dict["Manipulation"][f'{j}th'][f'SR_{manip_thr}'] = tmp_sr * 100
            
            print(f"Manip relative threshold: {manip_thr}")
            print(f"\t{' -> '.join(closed_loop_sr)}")
            
    def summary_manip(self):
        print("\n************** Manipulation Stage Analysis **************")
        self.summary_manip_helper()
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
                        cfgs_path="/home/Programs/AKM/cfgs",
                        run_name=run_name,          
                        obj_idx=obj_idx,         
                        range_transition=f"{i}_{j}"
                    )
                )
    dataAnalyze = DataAnalyze(datalist=data_list)
    print_path = f"/home/Programs/AKM/assets/analysis/{run_name}.txt"
    with open(print_path, "w") as f:
        sys.stdout = f
        dataAnalyze.summary_explore()
        dataAnalyze.summary_recon()
        dataAnalyze.summary_manip()
    
    save_path = f"/home/Programs/AKM/assets/analysis/{run_name}.npy"
    np.save(save_path, dataAnalyze.processed_data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name')
    
    args = parser.parse_args()
    analyze(args.run_name)