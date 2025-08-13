import os
from tqdm import tqdm
import pickle
import sys
from PIL import Image
sys.path.append("/home/zby/Programs/AKM/third_party/RAM_code")

from subset_retrieval.subset_retrieve_pipeline import SubsetRetrievePipeline
from vision.featurizer.run_featurizer import extract_ft


class PreExtractDiftFeat(SubsetRetrievePipeline):
    def __init__(self, subset_dir, save_root=None, topk=5, lang_mode='clip', crop=True) -> None:
        super().__init__(subset_dir, save_root, topk, lang_mode, crop)
        
    def pre_extract(self):
        for data_source, tasks in self.task_dict.items():
            for task in tasks:
                task_dir = os.path.join(self.subset_dir, data_source, task.replace(" ", "_"))
                read_file_name = task.replace(" ", "_") + "_new.pkl"
                complete_path = os.path.join(task_dir, read_file_name)
                
                save_file_name = task.replace(" ", "_") + "_new_sd.pkl"
                save_path = os.path.join(task_dir, save_file_name)
                
                if not os.path.exists(complete_path):
                    assert f"{complete_path} not exists during feat pre_extraction!"
                elif os.path.exists(save_path):
                    print(f"{save_file_name} already exists, skip to next task..")
                    continue
                else:
                    print(f"[{data_source}/{task}] Start extracting..")
                    with open(complete_path, 'rb') as f:
                        data_dict = pickle.load(f)
                        data_dict["feat"] = []
                        # 开始提取特征
                        prompt = "A photo of " + task.split(" ")[-1]
                        for idx in tqdm(range(len(data_dict["img"]))):
                            feat = extract_ft(Image.fromarray(data_dict['masked_img'][idx]).convert("RGB"), prompt=prompt, ftype='sd')
                            data_dict["feat"].append(feat)
                        # save 新的 data_dict 到 save_path 中去
                        with open(save_path, 'wb') as save_file:
                            pickle.dump(data_dict, save_file)
                        print(f"Features saved to {save_path}")

if __name__ == "__main__":
    my_class = PreExtractDiftFeat(subset_dir="/home/zby/Programs/AKM/assets/RAM_memory")
    my_class.pre_extract()
    