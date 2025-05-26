import argparse
import json
# 改为从 cfgs 中读取变量的方式
from embodied_analogy.environment.manipulate_env import ManipulateEnv
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

## 用 argparse 读取存储路径, 保存方式为 save_folder 下保存 0.npy, 1.npy, ...

def test(test_cfgs):
    for idx, cfg in test_cfgs.items():
        manipulateEnv = ManipulateEnv(cfg=cfg)
        overall_result = manipulateEnv.main()
        manipulateEnv.delete()
        save_path = ""
        