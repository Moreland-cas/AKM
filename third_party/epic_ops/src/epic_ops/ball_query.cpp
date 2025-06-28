#include <torch/library.h>
#include "epic_ops/ball_query.h"

namespace epic_ops::ball_query {

std::tuple<at::Tensor, at::Tensor> ball_query(
    const at::Tensor& points,
    const at::Tensor& query,
    const at::Tensor& batch_indices,
    const at::Tensor& batch_offsets,
    double radius,
    int64_t num_samples,
    const c10::optional<at::Tensor>& point_labels,
    const c10::optional<at::Tensor>& query_labels) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::ball_query", "")
                       .typed<decltype(ball_query)>();
  return op.call(
      points, query, batch_indices, batch_offsets, radius, num_samples,
      point_labels, query_labels);
}

TORCH_LIBRARY_FRAGMENT(epic_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::ball_query(Tensor points, Tensor query, Tensor batch_indices, "
      "Tensor batch_offsets, float radius, int num_samples, Tensor? point_labels=None, "
      "Tensor? query_labels=None) -> (Tensor, Tensor)"));
}

} // namespace epic_ops::ball_query
