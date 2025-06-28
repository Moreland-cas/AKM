import torch


@torch.no_grad()
def nms(
    ious: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    ious = ious.contiguous()
    scores = scores.contiguous()

    return torch.ops.epic_ops.nms(ious, scores, threshold)
