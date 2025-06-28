#pragma once

#include <torch/types.h>

namespace epic_ops::reduce {

// 0: sum; 1: min; 2: max
at::Tensor segmented_reduce(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode);

std::tuple<at::Tensor, at::Tensor> segmented_maxpool(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end);

} // namespace epic_ops::reduce
