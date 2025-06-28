#include <torch/library.h>
#include "epic_ops/nms.h"

namespace epic_ops::nms {

at::Tensor nms(const at::Tensor& ious, const at::Tensor& scores, double threshold) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::nms", "")
                       .typed<decltype(nms)>();
  return op.call(ious, scores, threshold);
}

TORCH_LIBRARY_FRAGMENT(epic_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::nms(Tensor ious, Tensor scores, float threshold) -> Tensor"));
}

} // namespace epic_ops::nms
