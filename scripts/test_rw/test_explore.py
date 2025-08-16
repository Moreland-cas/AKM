
import yaml
from akm.realworld_envs.explore_env import ExploreEnv

if __name__ == "__main__":
    choices = ["cabinet", "drawer"]
    choice = choices[0]
    with open(f"/home/zby/Programs/AKM/cfgs/realworld_cfgs/{choice}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    exploreEnv = ExploreEnv(cfg=cfg)
    exploreEnv.main()
    