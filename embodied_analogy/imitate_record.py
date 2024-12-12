from embodied_analogy.base_env import BaseEnv

class ImitateEvc(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
        ):
        super().__init__(phy_timestep)
        self.setup_camera()
        
    def imitate_from_record(self):
        self.load_articulated_object(index=100051)

if __name__ == '__main__':
    demo = ImitateEvc()
    demo.imitate_from_record()
    