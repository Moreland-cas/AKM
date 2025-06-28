#include <torch/library.h>
#include "epic_ops/iou.h"

namespace epic_ops::iou {

at::Tensor instance_seg_iou_csr(
    const at::Tensor& proposal_indices,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::instance_seg_iou_csr", "")
                       .typed<decltype(instance_seg_iou_csr)>();
  return op.call(proposal_indices, instance_labels, num_points_per_instance);
}

at::Tensor instance_seg_iou(
    const at::Tensor& proposal_indices_begin,
    const at::Tensor& proposal_indices_end,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::instance_seg_iou", "")
                       .typed<decltype(instance_seg_iou)>();
  return op.call(proposal_indices_begin, proposal_indices_end, instance_labels, num_points_per_instance);
}

at::Tensor batch_instance_seg_iou(
    const at::Tensor& proposal_offsets,
    const at::Tensor& instance_labels,
    const at::Tensor& batch_indices,
    const at::Tensor& num_points_per_instance) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("epic_ops::batch_instance_seg_iou", "")
                       .typed<decltype(batch_instance_seg_iou)>();
  return op.call(
      proposal_offsets, instance_labels, batch_indices, num_points_per_instance);
}

TORCH_LIBRARY_FRAGMENT(epic_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::instance_seg_iou_csr(Tensor proposal_indices, Tensor instance_labels, "
      "Tensor num_points_per_instance) -> Tensor"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::instance_seg_iou(Tensor proposal_indices_begin, Tensor proposal_indices_end, "
      "Tensor instance_labels, Tensor num_points_per_instance) -> Tensor"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "epic_ops::batch_instance_seg_iou(Tensor proposal_offsets, Tensor instance_labels, "
      "Tensor batch_indices, Tensor num_points_per_instance) -> Tensor"));
}

} // namespace epic_ops::iou
