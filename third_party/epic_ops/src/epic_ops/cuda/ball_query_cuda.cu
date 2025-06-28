#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "epic_ops/ball_query.h"
#include "epic_ops/utils/thrust_allocator.h"

namespace epic_ops::ball_query {

template <typename scalar_t, typename index_t, bool use_label, typename policy_t>
void ball_query_cuda_impl_thrust(
    const policy_t& policy,
    index_t* __restrict__ indices_ptr,
    index_t* __restrict__ num_points_per_query_ptr,
    const scalar_t* const __restrict__ points_ptr,
    const scalar_t* const __restrict__ query_ptr,
    const index_t* const __restrict__ batch_indices_ptr,
    const index_t* const __restrict__ batch_offsets_ptr,
    const index_t* const __restrict__ point_labels_ptr,
    const index_t* const __restrict__ query_labels_ptr,
    scalar_t radius2,
    index_t num_samples,
    index_t num_queries) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_queries),
      [=] __host__ __device__ (index_t i) {
        const auto batch_idx = batch_indices_ptr[i];
        const auto begin = batch_offsets_ptr[batch_idx];
        const auto end = batch_offsets_ptr[batch_idx + 1];

        const auto q_label = use_label ? query_labels_ptr[i] : -1;
        const auto q_x = query_ptr[i * 3 + 0];
        const auto q_y = query_ptr[i * 3 + 1];
        const auto q_z = query_ptr[i * 3 + 2];

        index_t cnt = 0;
        for (auto k = begin; k < end && cnt < num_samples; k++) {
          const auto label = use_label ? point_labels_ptr[k] : -1;
          if (label != q_label) {
            continue;
          }

          const auto x = points_ptr[k * 3 + 0];
          const auto y = points_ptr[k * 3 + 1];
          const auto z = points_ptr[k * 3 + 2];
          const auto d2 = (q_x - x) * (q_x - x) + (q_y - y) * (q_y - y) +
                          (q_z - z) * (q_z - z);
          if (d2 < radius2) {
            indices_ptr[i * num_samples + cnt] = k;
            cnt++;
          }
        }
        num_points_per_query_ptr[i] = cnt;
      });
}

template <typename scalar_t, typename index_t>
void ball_query_cuda_impl(
    at::Tensor& indices,
    at::Tensor& num_points_per_query,
    const at::Tensor& points,
    const at::Tensor& query,
    const at::Tensor& batch_indices,
    const at::Tensor& batch_offsets,
    double radius,
    int64_t num_samples,
    const c10::optional<at::Tensor>& point_labels,
    const c10::optional<at::Tensor>& query_labels) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto num_queries = points.size(0);

  auto indices_ptr = indices.data_ptr<index_t>();
  auto num_points_per_query_ptr = num_points_per_query.data_ptr<index_t>();
  auto points_ptr = points.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto batch_indices_ptr = batch_indices.data_ptr<index_t>();
  auto batch_offsets_ptr = batch_offsets.data_ptr<index_t>();

  index_t* point_labels_ptr = nullptr;
  if (point_labels.has_value()) {
    point_labels_ptr = point_labels.value().data_ptr<index_t>();
  }
  index_t* query_labels_ptr = nullptr;
  if (query_labels.has_value()) {
    query_labels_ptr = query_labels.value().data_ptr<index_t>();
  }

  if (point_labels.has_value()) {
    ball_query_cuda_impl_thrust<scalar_t, index_t, true>(
        policy,
        indices_ptr,
        num_points_per_query_ptr,
        points_ptr,
        query_ptr,
        batch_indices_ptr,
        batch_offsets_ptr,
        point_labels_ptr,
        query_labels_ptr,
        static_cast<scalar_t>(radius * radius),
        static_cast<index_t>(num_samples),
        static_cast<index_t>(num_queries));
  } else {
    ball_query_cuda_impl_thrust<scalar_t, index_t, false>(
        policy,
        indices_ptr,
        num_points_per_query_ptr,
        points_ptr,
        query_ptr,
        batch_indices_ptr,
        batch_offsets_ptr,
        point_labels_ptr,
        query_labels_ptr,
        static_cast<scalar_t>(radius * radius),
        static_cast<index_t>(num_samples),
        static_cast<index_t>(num_queries));
  }

}

std::tuple<at::Tensor, at::Tensor> ball_query_cuda(
    const at::Tensor& points,
    const at::Tensor& query,
    const at::Tensor& batch_indices,
    const at::Tensor& batch_offsets,
    double radius,
    int64_t num_samples,
    const c10::optional<at::Tensor>& point_labels,
    const c10::optional<at::Tensor>& query_labels) {
  TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(batch_indices.is_cuda(), "batch_indices must be a CUDA tensor");
  TORCH_CHECK(batch_offsets.is_cuda(), "batch_offsets must be a CUDA tensor");

  TORCH_CHECK(points.dim() == 2, "points must be a 2D tensor");
  TORCH_CHECK(query.dim() == 2, "edges must be a 2D tensor");
  TORCH_CHECK(batch_indices.dim() == 1, "batch_indices must be a 1D tensor");
  TORCH_CHECK(batch_offsets.dim() == 1, "batch_offsets must be a 1D tensor");

  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");
  TORCH_CHECK(batch_offsets.is_contiguous(), "batch_offsets must be contiguous");

  if (point_labels.has_value()) {
    TORCH_CHECK(point_labels.value().is_cuda(), "point_labels must be a CUDA tensor");
    TORCH_CHECK(point_labels.value().dim() == 1, "point_labels must be a 1D tensor");
    TORCH_CHECK(point_labels.value().is_contiguous(), "point_labels must be contiguous");
  }

  if (query_labels.has_value()) {
    TORCH_CHECK(query_labels.value().is_cuda(), "query_labels must be a CUDA tensor");
    TORCH_CHECK(query_labels.value().dim() == 1, "query_labels must be a 1D tensor");
    TORCH_CHECK(query_labels.value().is_contiguous(), "query_labels must be contiguous");
  }

  auto indices = at::empty({query.size(0), num_samples}, batch_indices.options());
  auto num_points_per_query = at::empty({query.size(0)}, batch_indices.options());

  AT_DISPATCH_FLOATING_TYPES(points.type(), "ball_query_cuda", [&] {
    if (batch_indices.scalar_type() == at::kInt) {
      ball_query_cuda_impl<scalar_t, int32_t>(
          indices,
          num_points_per_query,
          points,
          query,
          batch_indices,
          batch_offsets,
          radius,
          num_samples,
          point_labels,
          query_labels);
    } else if (batch_indices.scalar_type() == at::kLong) {
      ball_query_cuda_impl<scalar_t, int64_t>(
          indices,
          num_points_per_query,
          points,
          query,
          batch_indices,
          batch_offsets,
          radius,
          num_samples,
          point_labels,
          query_labels);
    } else {
      AT_ERROR("Unsupported type (ball_query_cuda)");
    }
  });

  return {indices, num_points_per_query};
}

TORCH_LIBRARY_IMPL(epic_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::ball_query"),
         TORCH_FN(ball_query_cuda));
}

} // namespace epic_ops::ball_query