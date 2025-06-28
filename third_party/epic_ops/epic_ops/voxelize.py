from typing import List, Tuple
from contextlib import redirect_stdout
import io

import torch

with redirect_stdout(io.StringIO()):
    import open3d.ml.torch as ml3d

from .expand import expand_csr


def voxelize_raw(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    batch_offsets: torch.Tensor,
    voxel_size: List[float],
    points_range_min: List[float],
    points_range_max: List[float],
    reduction: str = "mean",
    max_points_per_voxel: int = 9223372036854775807,
    max_voxels: int = 9223372036854775807,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    with torch.no_grad():
        (
            voxel_coords, voxel_point_indices,
            voxel_point_row_splits, voxel_batch_splits,
        ) = torch.ops.open3d.voxelize(
            points,
            row_splits=batch_offsets,
            voxel_size=torch.as_tensor(voxel_size, dtype=torch.float32),
            points_range_min=torch.as_tensor(points_range_min, dtype=torch.float32),
            points_range_max=torch.as_tensor(points_range_max, dtype=torch.float32),
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        ) # type: ignore
        

        batch_indices, _ = expand_csr(voxel_batch_splits, voxel_coords.shape[0])

        voxel_indices, num_points_per_voxel = expand_csr(
            voxel_point_row_splits, voxel_point_indices.shape[0]
        )

        if voxel_point_indices.shape[0] == num_points:
            pc_voxel_id = torch.empty_like(voxel_point_indices)
        else:
            pc_voxel_id = torch.full(
                (num_points,), -1,
                dtype=voxel_point_indices.dtype, device=voxel_point_indices.device
            )
        pc_voxel_id.scatter_(dim=0, index=voxel_point_indices, src=voxel_indices)

    pt_features = pt_features[voxel_point_indices]
    voxel_features = torch.segment_reduce(pt_features, reduction, lengths=num_points_per_voxel)

    return voxel_features, voxel_coords, batch_indices, pc_voxel_id


def voxelize(
    points: torch.Tensor,
    pt_features: torch.Tensor,
    batch_offsets: torch.Tensor,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    reduction: str = "mean",
    max_points_per_voxel: int = 9223372036854775807,
    max_voxels: int = 9223372036854775807,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    with torch.no_grad():
        (voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits,) = torch.ops.open3d.voxelize(
            points,row_splits=batch_offsets,
            voxel_size=voxel_size,
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels
        )
        batch_indices, _ = expand_csr(voxel_batch_splits, voxel_coords.shape[0])

        voxel_indices, num_points_per_voxel = expand_csr(voxel_point_row_splits, voxel_point_indices.shape[0])

        if voxel_point_indices.shape[0] == num_points:
            pc_voxel_id = torch.empty_like(voxel_point_indices)
        else:
            pc_voxel_id = torch.full(
                (num_points,), -1,
                dtype=voxel_point_indices.dtype, device=voxel_point_indices.device
            )
        pc_voxel_id.scatter_(dim=0, index=voxel_point_indices, src=voxel_indices)

    # pdb.set_trace()
    pt_features = pt_features[voxel_point_indices]
    voxel_features = torch.segment_reduce(pt_features, reduction, lengths=num_points_per_voxel)

    return voxel_features, voxel_coords, batch_indices, pc_voxel_id

def test_1():
    points = torch.as_tensor([
        # batch #1
        [0.3, 0.3, 0],
        #
        [2.1, 2.2, 0],
        [2.6, 2.7, 0],
        [2.9, 2.1, 0],
        #
        [3.5, 4.9, 0],
        [3.2, 4.5, 0],
        #
        [4.9, 0.1, 0],
        [4.5, 0.6, 0],
        # batch #2
        [0.1, 2.7, 0],
        [0.6, 2.1, 0],
        #
        [1.9, 4.5, 0],
        #
        [3.2, 1.1, 0],
        #
        [4.5, 3.2, 0],
        [4.8, 3.1, 0],
        [4.2, 3.9, 0],
    ], dtype=torch.float32)

    voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits = ml3d.ops.voxelize(
        points,
        row_splits=torch.as_tensor([0, 8, 15], dtype=torch.int64),
        voxel_size=torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        points_range_min=torch.as_tensor([0, 0, 0], dtype=torch.float32),
        points_range_max=torch.as_tensor([5, 5, 5], dtype=torch.float32),
    )

    print("voxel_coords", voxel_coords)
    print("voxel_point_indices", voxel_point_indices)
    print("voxel_point_row_splits", voxel_point_row_splits)
    print("voxel_batch_splits", voxel_batch_splits)

    num_points_per_voxel = voxel_point_row_splits[1:] - voxel_point_row_splits[:-1]
    voxel_features = torch.segment_reduce(points, reduce="mean", lengths=num_points_per_voxel)
    print("voxel_features", voxel_features.shape)


def test_2():
    points = torch.as_tensor([
        # batch #1
        [0.3, 0.3, 0],
        #
        [2.1, 2.2, 0],
        [2.6, 2.7, 0],
        [2.9, 2.1, 0],
        #
        [3.5, 4.9, 0],
        [3.2, 4.5, 0],
        #
        [4.9, 0.1, 0],
        [4.5, 0.6, 0],
        # batch #2
        [0.1, 2.7, 0],
        [0.6, 2.1, 0],
        #
        [1.9, 4.5, 0],
        #
        [3.2, 1.1, 0],
        #
        [4.5, 3.2, 0],
        [4.8, 3.1, 0],
        [4.2, 3.9, 0],
    ], dtype=torch.float32, device="cuda")
    pt_features = torch.randn(points.shape[0], 16, dtype=torch.float32, device="cuda")

    voxel_features, voxel_coords, pc_voxel_id = voxelize(
        points,
        pt_features,
        batch_offsets_csr=torch.as_tensor([0, 8, 15], dtype=torch.int64, device="cuda"), # type: ignore
        voxel_size=torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        points_range_min=torch.as_tensor([0, 0, 0], dtype=torch.float32),
        points_range_max=torch.as_tensor([5, 5, 5], dtype=torch.float32),
    )

    print("voxel_features", voxel_features.shape)
    print("voxel_coords", voxel_coords.shape)
    print("voxel_coords", voxel_coords)
    print("pc_voxel_id", pc_voxel_id)

    pt_coords = voxel_coords[:, 1:][pc_voxel_id]
    print("pt_coords", pt_coords)
    print((pt_coords == points.long()).all())


if __name__ == "__main__":
    test_2()
