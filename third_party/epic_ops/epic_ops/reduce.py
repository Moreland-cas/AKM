from typing import Tuple

import torch


@torch.no_grad()
def segmented_reduce(
    values: torch.Tensor,
    segment_offsets_begin: torch.Tensor,
    segment_offsets_end: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    if mode == "sum":
        mode_id = 0
    elif mode == "min":
        mode_id = 1
    elif mode == "max":
        mode_id = 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.ops.epic_ops.segmented_reduce(
        values, segment_offsets_begin, segment_offsets_end, mode_id
    )


def segmented_maxpool(
    values: torch.Tensor,
    segment_offsets_begin: torch.Tensor,
    segment_offsets_end: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    return torch.ops.epic_ops.segmented_maxpool(
        values, segment_offsets_begin, segment_offsets_end
    )


def test():
    import random

    num_segments = 128

    values = []
    segment_offsets = [0]
    for i in range(num_segments):
        num_values_per_seg = random.randint(1, 1023)
        values.append(torch.randn(num_values_per_seg, 3, dtype=torch.float64, device="cuda"))
        segment_offsets.append(segment_offsets[-1] + num_values_per_seg)

    values = torch.cat(values, dim=0)
    segment_offsets_begin = torch.tensor(segment_offsets[:-1], dtype=torch.int32, device="cuda")
    segment_offsets_end = torch.tensor(segment_offsets[1:], dtype=torch.int32, device="cuda")

    result = segmented_reduce(values, segment_offsets_begin, segment_offsets_end, mode="sum")

    for i in range(num_segments):
        values_per_seg = values[segment_offsets[i]:segment_offsets[i + 1]]
        assert torch.allclose(result[i], values_per_seg.sum(0)), i

    print("done")


def test_2():
    a1 = torch.as_tensor([
        #
        [1, 2, 3],
        [0, 7, 4],
        [3, 5, 1],
        #
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=torch.float32, device="cuda")
    a1.requires_grad_(True)

    a2 = torch.as_tensor([
        #
        [1, 2, 3],
        [0, 7, 4],
        [3, 5, 1],
        #
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=torch.float32, device="cuda")
    a2.requires_grad_(True)

    c, _ = segmented_maxpool(
        a1,
        torch.as_tensor([0, 3], dtype=torch.int32).cuda(),
        torch.as_tensor([3, 5], dtype=torch.int32).cuda(),
    )

    L1 = c.mean()

    L1.backward()

    print(a1.grad)


if __name__ == "__main__":
    # test()
    test_2()
