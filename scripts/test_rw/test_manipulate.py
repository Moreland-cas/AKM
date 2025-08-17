import yaml
from akm.realworld_envs.manipulate_env import ManipulateEnv

if __name__ == "__main__":
    choices = ["cabinet", "drawer"]
    choice = choices[0]
    
    with open(f"/home/Programs/AKM/cfgs/realworld_cfgs/{choice}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    manipEnv = ManipulateEnv(cfg=cfg)
    manipEnv.main()
    manipEnv.reset_robot_safe()