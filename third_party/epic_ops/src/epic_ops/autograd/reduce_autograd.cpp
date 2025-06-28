#include <torch/autograd.h>

#include "epic_ops/reduce.h"

namespace epic_ops::reduce {

class SegmentedMaxPoolFunction : public torch::autograd::Function<SegmentedMaxPoolFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& values,
      const torch::autograd::Variable& segment_offsets_begin,
      const torch::autograd::Variable& segment_offsets_end) {
    auto num_values = values.size(0);
    auto num_channels = values.size(1);

    ctx->saved_data["num_values"] = num_values;
    ctx->saved_data["num_channels"] = num_channels;

    at::AutoDispatchBelowADInplaceOrView g;
    auto [pool_values, max_indices] = segmented_maxpool(
        values, segment_offsets_begin, segment_offsets_end);

    ctx->save_for_backward({max_indices});

    return {pool_values, max_indices};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    const auto& saved_tensors = ctx->get_saved_variables();
    auto max_indices = saved_tensors[0];

    auto grad_output = grad_outputs[0].contiguous();

    auto num_values = ctx->saved_data["num_values"].toInt();
    auto num_channels = ctx->saved_data["num_channels"].toInt();

    auto grad_values = torch::zeros(
        {num_values, num_channels}, grad_output.options());
    grad_values.scatter_(0, max_indices.toType(at::kLong), grad_output);

    return {grad_values, at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> segmented_maxpool_autograd(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end) {
  auto results = SegmentedMaxPoolFunction::apply(
      values, segment_offsets_begin, segment_offsets_end);
  return {results[0], results[1]};
}

TORCH_LIBRARY_IMPL(epic_ops, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::segmented_maxpool"),
         TORCH_FN(segmented_maxpool_autograd));
}

}
