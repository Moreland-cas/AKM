#pragma once

#include <torch/types.h>

namespace epic_ops::ccl {

at::Tensor connected_components_labeling(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted);

} // namespace epic_ops::ccl
