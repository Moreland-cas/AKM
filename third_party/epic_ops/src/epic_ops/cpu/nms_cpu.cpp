#include "epic_ops/nms.h"

namespace epic_ops::nms {

template <typename scalar_t>
void nms_cpu_impl(
    at::Tensor& keep,
    const at::Tensor& ious,
    const at::Tensor& scores,
    float threshold) {
  int num_proposals = ious.size(0);

  auto keep_ptr = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < num_proposals; i++) {
    keep_ptr[num_to_keep++] = i;
  }
}

at::Tensor nms_cpu(const at::Tensor& ious, const at::Tensor& scores, double threshold) {
  TORCH_CHECK(ious.is_cpu(), "ious must be a CPU tensor");
  TORCH_CHECK(ious.dim() == 2, "ious must be a 2D tensor");
  TORCH_CHECK(ious.is_contiguous(), "ious must be contiguous");

  TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(scores.dim() == 1, "scores must be a 1D tensor");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");

  auto keep = at::empty({ious.size(0)}, ious.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ious.scalar_type(), "nms_cpu", [&] {
    nms_cpu_impl<scalar_t>(keep, ious, scores, threshold);
  });

  return keep;
}

TORCH_LIBRARY_IMPL(epic_ops, CPU, m) {
  // m.impl(TORCH_SELECTIVE_NAME("epic_ops::nms"),
  //        TORCH_FN(nms_cpu));
}

} // namespace epic_ops::nms
