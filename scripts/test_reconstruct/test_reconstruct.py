from embodied_analogy.utility.utils import initialize_napari
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr

    
if __name__ == "__main__":
    cfg = {
        "num_kframes": 5,
        "obj_description": "cabinet",
        "fine_lr": 1e-3,
    }
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/logs_complex/test_explore_4_11/45168_1_revolute/explore/obj_repr.npy")
    obj_repr.reconstruct(
        # num_initial_pts=self.num_initial_pts,
        num_kframes=cfg["num_kframes"],
        obj_description=cfg["obj_description"],
        fine_lr=cfg["fine_lr"],
        file_path=None,
        evaluate=True,
        visualize=True,
    )