import torch


@torch.no_grad()
def instance_seg_iou_csr(
    proposal_indices: torch.Tensor,
    instance_labels: torch.Tensor,
    num_points_per_instance: torch.Tensor,
) -> torch.Tensor:
    proposal_indices = proposal_indices.contiguous()
    instance_labels = instance_labels.contiguous()
    num_points_per_instance = num_points_per_instance.contiguous()

    return torch.ops.epic_ops.instance_seg_iou_csr(
        proposal_indices, instance_labels, num_points_per_instance
    )


@torch.no_grad()
def instance_seg_iou(
    proposal_indices_begin: torch.Tensor,
    proposal_indices_end: torch.Tensor,
    instance_labels: torch.Tensor,
    num_points_per_instance: torch.Tensor,
) -> torch.Tensor:
    proposal_indices_begin = proposal_indices_begin.contiguous()
    proposal_indices_end = proposal_indices_end.contiguous()
    instance_labels = instance_labels.contiguous()
    num_points_per_instance = num_points_per_instance.contiguous()

    return torch.ops.epic_ops.instance_seg_iou(
        proposal_indices_begin, proposal_indices_end,
        instance_labels, num_points_per_instance,
    )


@torch.no_grad()
def batch_instance_seg_iou(
    proposal_offsets: torch.Tensor,
    instance_labels: torch.Tensor,
    batch_indices: torch.Tensor,
    num_points_per_instance: torch.Tensor,
) -> torch.Tensor:
    proposal_offsets = proposal_offsets.contiguous()
    instance_labels = instance_labels.contiguous()
    batch_indices = batch_indices.contiguous()
    num_points_per_instance = num_points_per_instance.contiguous()

    return torch.ops.epic_ops.batch_instance_seg_iou(
        proposal_offsets, instance_labels, batch_indices, num_points_per_instance
    )
