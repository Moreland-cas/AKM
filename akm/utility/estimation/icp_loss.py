import torch
import numpy as np

from akm.utility.utils import (
    compute_normals,
    find_correspondences
)


def icp_loss_torch(
    joint_type,
    Tref2tgt, 
    ref_pc, 
    tgt_pc, 
    target_normals, # better provided
    loss_type, # plane_to_plane
    icp_select_range, # 0.1
    num_sample_points=1000
    # num_sample_points=10000
):
    """
    Calculates ICP loss (point-to-point or point-to-plane), supporting translation or rotation.
    The only parameter that needs to be optimized is joint_dir. This is the only one that requires gradients.

    Parameters:
        joint_dir (torch.Tensor): shape (3,) , a learnable parameter with a modulus of 1
        ref_pc (np.ndarray): reference point cloud, shape (N, 3)
        tgt_pc (np.ndarray): target point cloud, shape (M, 3)
        target_normals (np.ndarray, optional): target point cloud normals, shape (M, 3), used only in point-to-plane mode
        loss_type (str): "point_to_point" or "point_to_plane"
        joint_type (str): "prismatic" or "revolute"
        sample_points (int): If the point cloud exceeds the limit, sample 10,000 points from it

    Returns:
        loss (torch.Tensor): the calculated loss value
    """
    assert joint_type in ["prismatic", "revolute"]
    assert Tref2tgt.is_cuda and "Tref2tgt tensor must be on GPU"
    assert Tref2tgt.requires_grad and "Tref2tgt tensor must requires grad to be optimized"
    assert loss_type in ["point_to_point", "point_to_plane"]
    
    # print(f"icp input len(ref_pc)={len(ref_pc)} len(tgt_pc)={len(tgt_pc)}")
    
    if min(len(ref_pc), len(tgt_pc)) < 100:
        return torch.tensor(0).cuda()

    if loss_type == "point_to_plane":
        assert target_normals is not None, "Although there is way to work around, it is recommended to provide target_normals, \
            since compute target normals based on sampled pointcloud here is less accurate"
        if target_normals is None:
            target_normals = compute_normals(tgt_pc) # N, 3
        target_normals = torch.tensor(target_normals, dtype=torch.float32, device=Tref2tgt.device)
        
    if len(ref_pc) > num_sample_points:
        sampled_index = np.random.choice(len(ref_pc), num_sample_points, replace=False)
        ref_pc = ref_pc[sampled_index]
    if len(tgt_pc) > num_sample_points:
        sampled_index = np.random.choice(len(tgt_pc), num_sample_points, replace=False)
        tgt_pc = tgt_pc[sampled_index]
        target_normals = target_normals[sampled_index]
    
    # print(f"downed sampled size len(ref_pc)={len(ref_pc)} len(tgt_pc)={len(tgt_pc)}")
    
    ref_pc = torch.tensor(ref_pc, dtype=torch.float32, device=Tref2tgt.device)
    target_pc = torch.tensor(tgt_pc, dtype=torch.float32, device=Tref2tgt.device)
    ref_pc_homogeneous = torch.cat([ref_pc, torch.ones(ref_pc.shape[0], 1, device=ref_pc.device)], dim=1)

    # apply transform
    transformed_ref_pc_homogeneous = ref_pc_homogeneous @ Tref2tgt.T 
    transformed_ref_pc = transformed_ref_pc_homogeneous[:, :3]  

    # NN search
    indices, _, valid_mask = find_correspondences(transformed_ref_pc.detach().cpu().numpy(), tgt_pc, max_distance=icp_select_range)
    
    if valid_mask.sum() < 100:
        return torch.tensor(0).cuda()

    matched_target_pc = target_pc[indices] # N, 3

    # compute loss
    if loss_type == "point_to_point":
        loss = torch.mean(torch.sum((transformed_ref_pc[valid_mask] - matched_target_pc[valid_mask]) ** 2, dim=1))
    elif loss_type == "point_to_plane":
        diffs = transformed_ref_pc - matched_target_pc
        matched_normals = target_normals[indices]
        proj_errors = torch.sum(diffs[valid_mask] * matched_normals[valid_mask], dim=1)
        loss = torch.mean(proj_errors ** 2)

    return loss


if __name__ == "__main__":
    def test_differentiable_icp_loss():
        ref_pc = np.random.rand(6000, 3).astype(np.float32) 
        tgt_pc = np.random.rand(8000, 3).astype(np.float32) 
        joint_dir = torch.tensor([1, 0.5, 1], requires_grad=True, device='cuda')

        loss = icp_loss_torch(
            joint_dir, ref_pc, tgt_pc,
            target_normals=None,
            loss_type="point_to_plane", 
            joint_type="prismatic", 
            coor_valid_distance=10
        )
        print(f"ICP Loss: {loss.item()}")
    test_differentiable_icp_loss()