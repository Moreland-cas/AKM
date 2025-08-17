import yaml
from akm.realworld_envs.reconstruct_env import ReconEnv

if __name__ == "__main__":
    choices = ["cabinet", "drawer"]
    choice = choices[0]
    
    with open(f"/home/Programs/AKM/cfgs/realworld_cfgs/{choice}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    reconEnv = ReconEnv(cfg=cfg)
    reconEnv.main()