from typing import Tuple

import torch


@torch.no_grad()
def expand_csr(
    offsets: torch.Tensor,
    output_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets.contiguous()

    return torch.ops.epic_ops.expand_csr(
        offsets, output_size
    )


def test():
    offsets = torch.as_tensor([
        # 0, 7, 9, 20
        0, 4, 8
    ], dtype=torch.int64, device="cuda")

    result, sizes = expand_csr(offsets, 20)
    print("result: ", result)
    print("sizes: ", sizes)


if __name__ == "__main__":
    test()
