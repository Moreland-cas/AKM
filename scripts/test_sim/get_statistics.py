import os
import math
import numpy as np
from typing import List, Tuple, Any, Dict, Union

def compute_stats(values: List[Union[int, float]]) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation (sample standard deviation) of a list of values
    """
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) >= 1 else 0.0
    std_dev = math.sqrt(variance)
    return mean, std_dev

def collect_values(dict_list: List[Dict[str, Any]], path: List[str]) -> List[Union[int, float]]:
    """
    Collect all values of the given path in the dictionary
    """
    values = []
    for d in dict_list:
        current = d
        for key in path:
            current = current[key]
        values.append(current)
    return values

def process_nested_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process nested dictionaries and calculate the mean and standard deviation of each path
    """
    if not dict_list:
        return {}
    
    template = dict_list[0]
    
    def recursive_build(current: Any, path: List[str]) -> Any:
        if isinstance(current, dict):
            new_dict = {}
            for key, value in current.items():
                new_path = path + [key]
                new_dict[key] = recursive_build(value, new_path)
            return new_dict
        elif isinstance(current, (int, float)):
            values = collect_values(dict_list, path)
            mean, std_dev = compute_stats(values)
            return {"mean": mean, "std": std_dev}
        else:
            raise ValueError("Unsupported type in dictionary")
    
    return recursive_build(template, [])


if __name__ == "__main__":
    run_names = [
        "ours_1",
        "ours_2",
        "ours_3",
        "ours_4",
        "ours_5",
        "ours_wo_IOR_1",
        "ours_wo_IOR_2",
        "ours_wo_IOR_3",
        "ours_wo_IOR_4",
        "ours_wo_IOR_5",
        "ours_wo_AT_1",
        "ours_wo_AT_2",
        "ours_wo_AT_3",
        "ours_wo_AT_4",
        "ours_wo_AT_5",
        "ours_wo_CA_1",
        "ours_wo_CA_2",
        "ours_wo_CA_3",
        "ours_wo_CA_4",
        "ours_wo_CA_5",
        "ours_zs_1",
        "ours_zs_2",
        "ours_zs_3",
        "ours_zs_4",
        "ours_zs_5",
        "gflow_1",
        "gflow_2",
        "gflow_3",
        "gflow_4",
        "gflow_5",
        "gpnet_1",
        "gpnet_2",
        "gpnet_3",
        "gpnet_4",
        "gpnet_5"
    ]
    result_path = "/home/zby/Programs/AKM/assets/analysis_batch"
    
    # For each category of experiments, calculate the average value of explore, reconstruct and manipulate. 
    # Note that explore also includes generalflow and gpnet.
    method_list = [
        "ours_",
        "ours_wo_IOR_",
        "ours_wo_AT_",
        "ours_wo_CA_",
        "ours_zs_",
        "gflow_",
        "gpnet_"
    ]
    for method in method_list:
        print("Method:", method)
        npy_list = []
        for idx in range(1, 6):
            data = np.load(os.path.join(result_path, f'{method}{idx}.npy'), allow_pickle=True).item()
            npy_list.append(data)
        print(process_nested_dict(npy_list))
        print()
         
    # explore statistic
    explore_list = [
        "ours_",
        "gpnet_",
        "gflow_"
    ]
    print("Explore summary:")
    for explore_method in explore_list:
        npy_list = []
        for idx in range(1, 6):
            data = np.load(os.path.join(result_path, f'{explore_method}{idx}.npy'), allow_pickle=True).item()
            npy_list.append(data)
    print()
    print(process_nested_dict(npy_list))
    