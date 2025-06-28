#include <torch/library.h>
#include "epic_ops/expand.h"

namespace epic_ops::expand {

std::tuple<at::Tensor, at::Tensor> expand_csr(const at::Tensor& offsets, int64_t output_size) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::expand_csr", "")
                       .typed<decltype(expand_csr)>();
  return op.call(offsets, output_size);
}

TORCH_LIBRARY_FRAGMENT(epic_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::expand_csr(Tensor offsets, int output_size) -> (Tensor, Tensor)"));
}

} // namespace epic_ops::expand
