#pragma once

#include <torch/types.h>

namespace epic_ops::ball_query {

std::tuple<at::Tensor, at::Tensor> ball_query(
    const at::Tensor& points,
    const at::Tensor& query,
    const at::Tensor& batch_indices,
    const at::Tensor& batch_offsets,
    double radius,
    int64_t num_samples,
    const c10::optional<at::Tensor>& point_labels,
    const c10::optional<at::Tensor>& query_labels);

} // namespace epic_ops::ball_query
