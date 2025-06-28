#include <vector>
#include <c10/cuda/CUDAGuard.h>

#include "epic_ops/nms.h"

namespace epic_ops::nms {

namespace {

const int kThreadsPerBlock = 64;

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

}

template <typename scalar_t>
__global__
void nms_cuda_kernel(
    int64_t* mask_ptr,
    int num_dets,
    const scalar_t* ious_ptr,
    const int64_t* sorted_indices_ptr,
    float threshold) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) {
    return;
  }

  const int row_size = min(num_dets - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size = min(num_dets - col_start * kThreadsPerBlock, kThreadsPerBlock);

  if (threadIdx.x < row_size) {
    const int row_idx = row_start * kThreadsPerBlock + threadIdx.x;
    const int det1_idx = sorted_indices_ptr[row_idx];
    const auto ious_row_ptr = ious_ptr + det1_idx * num_dets;

    uint64_t t = 0;
    for (int i = row_start == col_start ? threadIdx.x + 1 : 0; i < col_size; i++) {
      const int col_idx = col_start * kThreadsPerBlock + i;
      const int det2_idx = sorted_indices_ptr[col_idx];
      const auto iou = ious_row_ptr[det2_idx];
      if (iou > threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(num_dets, kThreadsPerBlock);
    mask_ptr[row_idx * col_blocks + col_start] = t;
  }
}

at::Tensor nms_cuda(const at::Tensor& ious, const at::Tensor& scores, double threshold) {
  TORCH_CHECK(ious.is_cuda(), "ious must be a CUDA tensor");
  TORCH_CHECK(ious.dim() == 2, "ious must be a 1D tensor");
  TORCH_CHECK(ious.is_contiguous(), "ious must be contiguous");

  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(scores.dim() == 1, "scores must be a 1D tensor");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");

  at::cuda::CUDAGuard device_guard(ious.device());

  int num_dets = ious.size(0);
  if (num_dets == 0) {
    return at::empty({0}, ious.options().dtype(at::kLong));
  }

  auto sorted_indices = std::get<1>(scores.sort(true, 0, true));

  const int col_blocks = ceil_div(num_dets, kThreadsPerBlock);

  auto mask = at::empty({num_dets * col_blocks}, ious.options().dtype(at::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(kThreadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ious.scalar_type(), "nms_kernel", [&] {
    nms_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        mask.data_ptr<int64_t>(),
        num_dets,
        ious.data_ptr<scalar_t>(),
        sorted_indices.data_ptr<int64_t>(),
        threshold);
  });

  auto mask_cpu = mask.to(at::kCPU);
  auto mask_cpu_ptr = mask_cpu.data_ptr<int64_t>();

  std::vector<uint64_t> remv(col_blocks, 0ULL);

  auto keep_cpu = at::empty(
      {num_dets}, ious.options().dtype(at::kLong).device(at::kCPU));
  auto keep_cpu_ptr = keep_cpu.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < num_dets; i++) {
    const int nblock = i / kThreadsPerBlock;
    const int inblock = i % kThreadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_cpu_ptr[num_to_keep++] = i;
      auto p = mask_cpu_ptr + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  auto keep = keep_cpu.narrow(0, 0, num_to_keep).to(ious.device());
  return sorted_indices.index({keep});
}

TORCH_LIBRARY_IMPL(epic_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::nms"),
         TORCH_FN(nms_cuda));
}

} // namespace epic_ops::nms
