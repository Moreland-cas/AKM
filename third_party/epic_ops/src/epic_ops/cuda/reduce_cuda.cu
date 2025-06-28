#include <limits>
#include <cub/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "epic_ops/reduce.h"
#include "epic_ops/utils/thrust_allocator.h"

namespace epic_ops::reduce {

template <typename scalar_t, int N>
struct array_t {
  scalar_t data[N];

  inline void fill(scalar_t x) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] = x;
    }
  }
};

template <typename scalar_t, int N>
struct array_sum_op {
  using value_type = array_t<scalar_t, N>;

  __forceinline__
  __host__ __device__
  value_type operator() (const value_type& a, const value_type& b) const {
    value_type res;
    #pragma unroll
    for (int i = 0; i < N; i++) {
      res.data[i] = a.data[i] + b.data[i];
    }
    return res;
  }
};

template <typename scalar_t, int N>
struct array_min_op {
  using value_type = array_t<scalar_t, N>;

  __forceinline__
  __host__ __device__
  value_type operator() (const value_type& a, const value_type& b) const {
    value_type res;
    #pragma unroll
    for (int i = 0; i < N; i++) {
      res.data[i] = a.data[i] < b.data[i] ? a.data[i] : b.data[i];
    }
    return res;
  }
};

template <typename scalar_t, int N>
struct array_max_op {
  using value_type = array_t<scalar_t, N>;

  __forceinline__
  __host__ __device__
  value_type operator() (const value_type& a, const value_type& b) const {
    value_type res;
    #pragma unroll
    for (int i = 0; i < N; i++) {
      res.data[i] = a.data[i] > b.data[i] ? a.data[i] : b.data[i];
    }
    return res;
  }
};

template <typename scalar_t, typename index_t, int num_channels>
void segmented_reduce_cuda_impl(
    at::Tensor& output,
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  auto num_segments = segment_offsets_begin.size(0);

  auto output_ptr = reinterpret_cast<array_t<scalar_t, num_channels>*>(
      output.data_ptr<scalar_t>());
  auto values_ptr = reinterpret_cast<array_t<scalar_t, num_channels>*>(
      values.data<scalar_t>());
  auto segment_offsets_begin_ptr = segment_offsets_begin.data<index_t>();
  auto segment_offsets_end_ptr = segment_offsets_end.data<index_t>();

  void* d_temp_storage = nullptr;
  size_t num_temp_storage_bytes = 0;
  array_t<scalar_t, num_channels> initial_value;

  if (mode == 0) {  // sum
    initial_value.fill(0);
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_sum_op<scalar_t, num_channels>(),
        initial_value);
  } else if (mode == 1) { // min
    initial_value.fill(std::numeric_limits<scalar_t>::max());
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_min_op<scalar_t, num_channels>(),
        initial_value);
  } else {  // max
    initial_value.fill(std::numeric_limits<scalar_t>::min());
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_max_op<scalar_t, num_channels>(),
        initial_value);
  }

  d_temp_storage = c10::cuda::CUDACachingAllocator::raw_alloc(num_temp_storage_bytes);

  if (mode == 0) {  // sum
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_sum_op<scalar_t, num_channels>(),
        initial_value);
  } else if (mode == 1) { // min
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_min_op<scalar_t, num_channels>(),
        initial_value);
  } else {  // max
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, num_temp_storage_bytes,
        values_ptr, output_ptr, num_segments,
        segment_offsets_begin_ptr, segment_offsets_end_ptr,
        array_max_op<scalar_t, num_channels>(),
        initial_value);
  }

  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

at::Tensor segmented_reduce_cuda(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(segment_offsets_begin.is_cuda(), "segment_offsets_begin must be a CUDA tensor");
  TORCH_CHECK(segment_offsets_end.is_cuda(), "segment_offsets_end must be a CUDA tensor");

  TORCH_CHECK(values.dim() == 2, "values must be a 2D tensor");
  TORCH_CHECK(segment_offsets_begin.dim() == 1, "segment_offsets_begin must be a 1D tensor");
  TORCH_CHECK(segment_offsets_end.dim() == 1, "segment_offsets_end must be a 1D tensor");
  TORCH_CHECK(segment_offsets_begin.size(0) == segment_offsets_end.size(0),
              "segment_offsets_begin and segment_offsets_end must have the same size");

  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(segment_offsets_begin.is_contiguous(), "segment_offsets_begin must be contiguous");
  TORCH_CHECK(segment_offsets_end.is_contiguous(), "segment_offsets_end must be contiguous");

  TORCH_CHECK(0 <= mode && mode <= 2, "mode must be in [0, 2]");

  auto num_segments = segment_offsets_begin.size(0);
  auto num_channels = values.size(1);
  TORCH_CHECK(1 <= num_channels && num_channels <= 4, "num_channels must be in [1, 4]");

  auto output = at::empty({num_segments, num_channels}, values.options());

  AT_DISPATCH_FLOATING_TYPES(values.type(), "segmented_reduce_cuda", [&] {
    if (segment_offsets_begin.scalar_type() == at::kInt) {
      if (num_channels == 1) {
        segmented_reduce_cuda_impl<scalar_t, int32_t, 1>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 2) {
        segmented_reduce_cuda_impl<scalar_t, int32_t, 2>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 3) {
        segmented_reduce_cuda_impl<scalar_t, int32_t, 3>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 4) {
        segmented_reduce_cuda_impl<scalar_t, int32_t, 4>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else {
        AT_ERROR("Unsupported number of channels");
      }
    } else if (segment_offsets_begin.scalar_type() == at::kLong) {
      if (num_channels == 1) {
        segmented_reduce_cuda_impl<scalar_t, int64_t, 1>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 2) {
        segmented_reduce_cuda_impl<scalar_t, int64_t, 2>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 3) {
        segmented_reduce_cuda_impl<scalar_t, int64_t, 3>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else if (num_channels == 4) {
        segmented_reduce_cuda_impl<scalar_t, int64_t, 4>(
            output, values, segment_offsets_begin, segment_offsets_end, mode);
      } else {
        AT_ERROR("Unsupported number of channels");
      }
    } else {
      AT_ERROR("Unsupported type (segmented_reduce_cuda)");
    }
  });

  return output;
}

template <typename scalar_t, typename index_t>
void segmented_maxpool_cuda_impl(
    at::Tensor& output,
    at::Tensor& max_indices,
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  index_t num_segments = segment_offsets_begin.size(0);
  index_t num_channels = values.size(1);

  auto output_ptr = output.data_ptr<scalar_t>();
  auto max_indices_ptr = max_indices.data_ptr<index_t>();
  auto values_ptr = values.data_ptr<scalar_t>();
  auto segment_offsets_begin_ptr = segment_offsets_begin.data_ptr<index_t>();
  auto segment_offsets_end_ptr = segment_offsets_end.data_ptr<index_t>();

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_segments * num_channels),
      [=] __host__ __device__ (index_t idx) {
        index_t segment_idx = idx / num_channels;
        index_t channel_idx = idx % num_channels;

        auto begin = segment_offsets_begin_ptr[segment_idx];
        auto end = segment_offsets_end_ptr[segment_idx];

        index_t max_idx = -1;
        scalar_t max_value = -std::numeric_limits<scalar_t>::max();

        for (auto i = begin; i < end; i++) {
          auto value = values_ptr[i * num_channels + channel_idx];
          if (value > max_value) {
            max_value = value;
            max_idx = i;
          }
        }

        output_ptr[idx] = max_value;
        max_indices_ptr[idx] = max_idx;
      });
}

std::tuple<at::Tensor, at::Tensor> segmented_maxpool_cuda(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end) {
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(segment_offsets_begin.is_cuda(), "segment_offsets_begin must be a CUDA tensor");
  TORCH_CHECK(segment_offsets_end.is_cuda(), "segment_offsets_end must be a CUDA tensor");

  TORCH_CHECK(values.dim() == 2, "values must be a 2D tensor");
  TORCH_CHECK(segment_offsets_begin.dim() == 1, "segment_offsets_begin must be a 1D tensor");
  TORCH_CHECK(segment_offsets_end.dim() == 1, "segment_offsets_end must be a 1D tensor");
  TORCH_CHECK(segment_offsets_begin.size(0) == segment_offsets_end.size(0),
              "segment_offsets_begin and segment_offsets_end must have the same size");

  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(segment_offsets_begin.is_contiguous(), "segment_offsets_begin must be contiguous");
  TORCH_CHECK(segment_offsets_end.is_contiguous(), "segment_offsets_end must be contiguous");

  auto num_segments = segment_offsets_begin.size(0);
  auto num_channels = values.size(1);

  auto output = at::empty({num_segments, num_channels}, values.options());
  auto max_indices = at::empty({num_segments, num_channels}, segment_offsets_begin.options());

  AT_DISPATCH_FLOATING_TYPES(values.type(), "segmented_maxpool_cuda", [&] {
    if (segment_offsets_begin.scalar_type() == at::kInt) {
      segmented_maxpool_cuda_impl<scalar_t, int32_t>(
          output, max_indices, values, segment_offsets_begin, segment_offsets_end);
    } else if (segment_offsets_begin.scalar_type() == at::kLong) {
      segmented_maxpool_cuda_impl<scalar_t, int64_t>(
          output, max_indices, values, segment_offsets_begin, segment_offsets_end);
    } else {
      AT_ERROR("Unsupported type (segmented_maxpool_cuda)");
    }
  });

  return {output, max_indices};
}

TORCH_LIBRARY_IMPL(epic_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::segmented_reduce"),
         TORCH_FN(segmented_reduce_cuda));

  m.impl(TORCH_SELECTIVE_NAME("epic_ops::segmented_maxpool"),
         TORCH_FN(segmented_maxpool_cuda));
}

} // namespace epic_ops::reduce
