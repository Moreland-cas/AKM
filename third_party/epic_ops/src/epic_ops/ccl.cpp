#include <torch/library.h>
#include "epic_ops/ccl.h"

namespace epic_ops::ccl {

at::Tensor connected_components_labeling(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::connected_components_labeling", "")
                       .typed<decltype(connected_components_labeling)>();
  return op.call(indices, edges, compacted);
}

TORCH_LIBRARY_FRAGMENT(epic_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::connected_components_labeling(Tensor indices, Tensor edges, "
      "bool compacted) -> Tensor"));
}

} // namespace epic_ops::ccl
