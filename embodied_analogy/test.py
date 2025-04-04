import json
from embodied_analogy.environment.manipulate_env import ManipulateEnv

# 从 JSON 文件加载数据
with open('./test.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    
print("testing performance on prismatic joints...")
for cfg in data["prismatic_test_cfgs"]:
    
    me = ManipulateEnv(
        base_cfg=cfg["base_cfg"],
        robot_cfg=cfg["robot_cfg"],
        explore_cfg=cfg["explore_cfg"],
        recon_cfg=cfg["recon_cfg"],
        manip_cfg=cfg["manip_cfg"],
        task_cfg=cfg["task_cfg"]
    )
    result = me.main()
    print(result)
    pass

print("testing performance on revolute joints...")