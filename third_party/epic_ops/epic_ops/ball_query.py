from typing import Tuple, Optional
from contextlib import redirect_stdout
import io

import torch

with redirect_stdout(io.StringIO()):
    import open3d.ml.torch as ml3d


@torch.no_grad()
def ball_query(
    points: torch.Tensor,
    query: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    radius: float,
    num_samples: int,
    point_labels: Optional[torch.Tensor] = None,
    query_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    points = points.contiguous()
    query = query.contiguous()
    batch_indices = batch_indices.contiguous()
    batch_offsets = batch_offsets.contiguous()

    if point_labels is not None:
        point_labels = point_labels.contiguous()

    if query_labels is not None:
        query_labels = query_labels.contiguous()

    return torch.ops.epic_ops.ball_query(
        points, query, batch_indices, batch_offsets, radius, num_samples,
        point_labels, query_labels,
    )


@torch.no_grad()
def ball_query_fast(
    points: torch.Tensor,
    points_batch_offsets: torch.Tensor,
    queries: torch.Tensor,
    queries_batch_offsets: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        hash_table_index, hash_table_cell_splits, hash_table_splits
    ) = torch.ops.open3d.build_spatial_hash_table(
        points=points,
        radius=radius,
        points_row_splits=points_batch_offsets,
        hash_table_size_factor=1 / 32,
        max_hash_table_size=33554432
    )

    (
        neighbors_index, neighbors_row_splits, _
    ) = torch.ops.open3d.fixed_radius_search(
        points=points,
        queries=queries,
        radius=radius,
        points_row_splits=points_batch_offsets,
        queries_row_splits=queries_batch_offsets,
        hash_table_splits=hash_table_splits,
        hash_table_index=hash_table_index,
        hash_table_cell_splits=hash_table_cell_splits,
        index_dtype=3,
        metric="L2",
        ignore_query_point=False,
        return_distances=False
    )

    return neighbors_index, neighbors_row_splits
