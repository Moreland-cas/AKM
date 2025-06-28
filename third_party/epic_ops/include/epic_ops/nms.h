#pragma once

#include <torch/types.h>

namespace epic_ops::nms {

at::Tensor nms(const at::Tensor& ious, const at::Tensor& scores, double threshold);

} // namespace epic_ops::nms
