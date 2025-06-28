#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "epic_ops/iou.h"
#include "epic_ops/utils/thrust_allocator.h"

namespace epic_ops::iou {

template <typename scalar_t, typename index_t>
void instance_seg_iou_csr_cuda_impl(
    at::Tensor& ious,
    const at::Tensor& proposal_indices,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  index_t num_proposals = proposal_indices.size(0) - 1;
  index_t num_instances = num_points_per_instance.size(0);

  auto ious_ptr = ious.data_ptr<scalar_t>();
  auto proposal_indices_ptr = proposal_indices.data_ptr<index_t>();
  auto instance_labels_ptr = instance_labels.data_ptr<index_t>();
  auto num_points_per_instance_ptr = num_points_per_instance.data_ptr<index_t>();

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_proposals * num_instances),
      [=] __host__ __device__ (index_t idx) {
        const index_t proposal_idx = idx / num_instances;
        const index_t instance_idx = idx % num_instances;
        const auto begin = proposal_indices_ptr[proposal_idx];
        const auto end = proposal_indices_ptr[proposal_idx + 1];
        const auto num_points_proposal = end - begin;
        const auto num_points_gt = num_points_per_instance_ptr[instance_idx];

        index_t intersection = 0;
        for (int i = begin; i < end; i++) {
          if (instance_labels_ptr[i] == instance_idx) {
            intersection++;
          }
        }
        ious_ptr[idx] = static_cast<scalar_t>(intersection) / (
            num_points_gt + num_points_proposal - intersection + 1e-8f);
      });
}

at::Tensor instance_seg_iou_csr_cuda(
    const at::Tensor& proposal_indices,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  TORCH_CHECK(proposal_indices.is_cuda(), "proposal_indices must be a CUDA tensor");
  TORCH_CHECK(instance_labels.is_cuda(), "instance_labels must be a CUDA tensor");
  TORCH_CHECK(num_points_per_instance.is_cuda(), "num_points_per_instance must be a CUDA tensor");

  TORCH_CHECK(proposal_indices.dim() == 1, "proposal_indices must be a 1D tensor");
  TORCH_CHECK(instance_labels.dim() == 1, "instance_labels must be a 1D tensor");
  TORCH_CHECK(num_points_per_instance.dim() == 1, "num_points_per_instance must be a 1D tensor");

  TORCH_CHECK(proposal_indices.is_contiguous(), "proposal_indices must be contiguous");
  TORCH_CHECK(instance_labels.is_contiguous(), "instance_labels must be contiguous");
  TORCH_CHECK(num_points_per_instance.is_contiguous(), "num_points_per_instance must be contiguous");

  auto num_proposals = proposal_indices.size(0) - 1;
  auto num_instances = num_points_per_instance.size(0);
  auto ious = at::empty({num_proposals, num_instances}, proposal_indices.options().dtype(at::kFloat));

  if (proposal_indices.scalar_type() == at::kInt) {
    instance_seg_iou_csr_cuda_impl<float, int32_t>(
        ious,
        proposal_indices,
        instance_labels,
        num_points_per_instance);
  } else if (proposal_indices.scalar_type() == at::kLong) {
    instance_seg_iou_csr_cuda_impl<float, int64_t>(
        ious,
        proposal_indices,
        instance_labels,
        num_points_per_instance);
  } else {
    AT_ERROR("Unsupported type (instance_seg_iou_csr_cuda)");
  }

  return ious;
}

template <typename scalar_t, typename index_t>
void instance_seg_iou_cuda_impl(
    at::Tensor& ious,
    const at::Tensor& proposal_indices_begin,
    const at::Tensor& proposal_indices_end,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  index_t num_proposals = proposal_indices_begin.size(0);
  index_t num_instances = num_points_per_instance.size(0);

  auto ious_ptr = ious.data_ptr<scalar_t>();
  auto proposal_indices_begin_ptr = proposal_indices_begin.data_ptr<index_t>();
  auto proposal_indices_end_ptr = proposal_indices_end.data_ptr<index_t>();
  auto instance_labels_ptr = instance_labels.data_ptr<index_t>();
  auto num_points_per_instance_ptr = num_points_per_instance.data_ptr<index_t>();

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_proposals * num_instances),
      [=] __host__ __device__ (index_t idx) {
        const index_t proposal_idx = idx / num_instances;
        const index_t instance_idx = idx % num_instances;
        const auto begin = proposal_indices_begin_ptr[proposal_idx];
        const auto end = proposal_indices_end_ptr[proposal_idx];
        const auto num_points_proposal = end - begin;
        const auto num_points_gt = num_points_per_instance_ptr[instance_idx];

        index_t intersection = 0;
        for (int i = begin; i < end; i++) {
          if (instance_labels_ptr[i] == instance_idx) {
            intersection++;
          }
        }
        ious_ptr[idx] = static_cast<scalar_t>(intersection) / (
            num_points_gt + num_points_proposal - intersection + 1e-8f);
      });
}

at::Tensor instance_seg_iou_cuda(
    const at::Tensor& proposal_indices_begin,
    const at::Tensor& proposal_indices_end,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance) {
  TORCH_CHECK(proposal_indices_begin.is_cuda(), "proposal_indices_begin must be a CUDA tensor");
  TORCH_CHECK(proposal_indices_end.is_cuda(), "proposal_indices_end must be a CUDA tensor");
  TORCH_CHECK(instance_labels.is_cuda(), "instance_labels must be a CUDA tensor");
  TORCH_CHECK(num_points_per_instance.is_cuda(), "num_points_per_instance must be a CUDA tensor");

  TORCH_CHECK(proposal_indices_begin.dim() == 1, "proposal_indices_begin must be a 1D tensor");
  TORCH_CHECK(proposal_indices_end.dim() == 1, "proposal_indices_end must be a 1D tensor");
  TORCH_CHECK(instance_labels.dim() == 1, "instance_labels must be a 1D tensor");
  TORCH_CHECK(num_points_per_instance.dim() == 1, "num_points_per_instance must be a 1D tensor");

  TORCH_CHECK(proposal_indices_begin.is_contiguous(), "proposal_indices_begin must be contiguous");
  TORCH_CHECK(proposal_indices_end.is_contiguous(), "proposal_indices_end must be contiguous");
  TORCH_CHECK(instance_labels.is_contiguous(), "instance_labels must be contiguous");
  TORCH_CHECK(num_points_per_instance.is_contiguous(), "num_points_per_instance must be contiguous");

  auto num_proposals = proposal_indices_begin.size(0);
  auto num_instances = num_points_per_instance.size(0);
  auto ious = at::empty({num_proposals, num_instances}, proposal_indices_begin.options().dtype(at::kFloat));

  if (proposal_indices_begin.scalar_type() == at::kInt) {
    instance_seg_iou_cuda_impl<float, int32_t>(
        ious,
        proposal_indices_begin,
        proposal_indices_end,
        instance_labels,
        num_points_per_instance);
  } else if (proposal_indices_begin.scalar_type() == at::kLong) {
    instance_seg_iou_cuda_impl<float, int64_t>(
        ious,
        proposal_indices_begin,
        proposal_indices_end,
        instance_labels,
        num_points_per_instance);
  } else {
    AT_ERROR("Unsupported type (instance_seg_iou_cuda)");
  }

  return ious;
}

template <typename scalar_t, typename index_t>
void batch_instance_seg_iou_cuda_impl(
    at::Tensor& ious,
    const at::Tensor& proposal_offsets,
    const at::Tensor& instance_labels,
    const at::Tensor& batch_indices,
    const at::Tensor& num_points_per_instance) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  const index_t num_proposals = proposal_offsets.size(0) - 1;
  const index_t max_num_instances = num_points_per_instance.size(1);

  auto ious_ptr = ious.data_ptr<scalar_t>();
  auto proposal_offsets_ptr = proposal_offsets.data_ptr<index_t>();
  auto instance_labels_ptr = instance_labels.data_ptr<index_t>();
  auto batch_indices_ptr = batch_indices.data_ptr<index_t>();
  auto num_points_per_instance_ptr = num_points_per_instance.data_ptr<index_t>();

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_proposals * max_num_instances),
      [=] __host__ __device__ (index_t idx) {
        const index_t proposal_idx = idx / max_num_instances;
        const index_t instance_idx = idx % max_num_instances;
        const auto begin = proposal_offsets_ptr[proposal_idx];
        const auto end = proposal_offsets_ptr[proposal_idx + 1];
        const auto num_points_proposal = end - begin;
        const auto batch_idx = batch_indices_ptr[begin];
        const auto num_points_gt = num_points_per_instance_ptr[
            batch_idx * max_num_instances + instance_idx];

        if (num_points_gt == 0 || num_points_proposal == 0) {
          ious_ptr[idx] = 0.;
          return;
        }

        index_t intersection = 0;
        for (int i = begin; i < end; i++) {
          auto instance_label = instance_labels_ptr[i];
          if (instance_label == instance_idx) {
            intersection++;
          }
        }
        ious_ptr[idx] = static_cast<scalar_t>(intersection) / (
            num_points_gt + num_points_proposal - intersection + 1e-8f);
      });
}

at::Tensor batch_instance_seg_iou_cuda(
    const at::Tensor& proposal_offsets,
    const at::Tensor& instance_labels,
    const at::Tensor& batch_indices,
    const at::Tensor& num_points_per_instance) {
  TORCH_CHECK(proposal_offsets.is_cuda(), "proposal_offsets must be a CUDA tensor");
  TORCH_CHECK(instance_labels.is_cuda(), "instance_labels must be a CUDA tensor");
  TORCH_CHECK(batch_indices.is_cuda(), "batch_indices must be a CUDA tensor");
  TORCH_CHECK(num_points_per_instance.is_cuda(), "num_points_per_instance must be a CUDA tensor");

  TORCH_CHECK(proposal_offsets.dim() == 1, "proposal_offsets must be a 1D tensor");
  TORCH_CHECK(instance_labels.dim() == 1, "instance_labels must be a 1D tensor");
  TORCH_CHECK(batch_indices.dim() == 1, "batch_indices must be a 1D tensor");
  TORCH_CHECK(num_points_per_instance.dim() == 2, "num_points_per_instance must be a 2D tensor");

  TORCH_CHECK(proposal_offsets.is_contiguous(), "proposal_offsets must be contiguous");
  TORCH_CHECK(instance_labels.is_contiguous(), "instance_labels must be contiguous");
  TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");
  TORCH_CHECK(num_points_per_instance.is_contiguous(), "num_points_per_instance must be contiguous");

  auto num_proposals = proposal_offsets.size(0) - 1;
  auto max_num_instances = num_points_per_instance.size(1);
  auto ious = at::empty(
      {num_proposals, max_num_instances}, proposal_offsets.options().dtype(at::kFloat));

  if (proposal_offsets.scalar_type() == at::kInt) {
    batch_instance_seg_iou_cuda_impl<float, int32_t>(
        ious,
        proposal_offsets,
        instance_labels,
        batch_indices,
        num_points_per_instance);
  } else if (proposal_offsets.scalar_type() == at::kLong) {
    batch_instance_seg_iou_cuda_impl<float, int64_t>(
        ious,
        proposal_offsets,
        instance_labels,
        batch_indices,
        num_points_per_instance);
  } else {
    AT_ERROR("Unsupported type (batch_instance_seg_iou_cuda)");
  }

  return ious;
}

TORCH_LIBRARY_IMPL(epic_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::instance_seg_iou_csr"),
         TORCH_FN(instance_seg_iou_csr_cuda));

  m.impl(TORCH_SELECTIVE_NAME("epic_ops::instance_seg_iou"),
         TORCH_FN(instance_seg_iou_cuda));

  m.impl(TORCH_SELECTIVE_NAME("epic_ops::batch_instance_seg_iou"),
         TORCH_FN(batch_instance_seg_iou_cuda));
}

} // namespace epic_ops::iou