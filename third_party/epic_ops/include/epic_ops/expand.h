#pragma once

#include <torch/types.h>

namespace epic_ops::expand {

std::tuple<at::Tensor, at::Tensor> expand_csr(
    const at::Tensor& offsets, int64_t output_size);

} // namespace epic_ops::expand
