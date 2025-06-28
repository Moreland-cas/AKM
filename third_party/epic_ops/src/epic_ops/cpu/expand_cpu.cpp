#include <limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "epic_ops/expand.h"

namespace epic_ops::expand {

template <typename index_t>
void expand_csr_cpu_impl(
    at::Tensor& output,
    at::Tensor& sizes,
    const at::Tensor& offsets,
    int64_t output_size) {
  index_t num_segments = offsets.size(0) - 1;

  auto output_ptr = output.data_ptr<index_t>();
  auto offsets_ptr = offsets.data_ptr<index_t>();
  auto sizes_ptr = sizes.data_ptr<index_t>();

  thrust::transform(
      thrust::host,
      thrust::make_zip_iterator(thrust::make_tuple(
          offsets_ptr, offsets_ptr + 1)),
      thrust::make_zip_iterator(thrust::make_tuple(
          offsets_ptr + num_segments, offsets_ptr + 1 + num_segments)),
      sizes_ptr,
      [=] __host__ (thrust::tuple<index_t, index_t> t) {
        return thrust::get<1>(t) - thrust::get<0>(t);
      });

  thrust::scatter_if(
      thrust::host,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_segments),
      offsets_ptr,
      sizes_ptr,
      output_ptr);

  thrust::inclusive_scan(
      thrust::host,
      output_ptr,
      output_ptr + output_size,
      output_ptr,
      thrust::maximum<index_t>());
}

std::tuple<at::Tensor, at::Tensor> expand_csr_cpu(const at::Tensor& offsets, int64_t output_size) {
  TORCH_CHECK(offsets.is_cpu(), "offsets must be a CPU tensor");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be a 1D tensor");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");

  auto num_segments = offsets.size(0) - 1;
  auto output = at::zeros({output_size}, offsets.options());
  auto sizes = at::empty({num_segments}, offsets.options());

  if (offsets.scalar_type() == at::kInt) {
    expand_csr_cpu_impl<int32_t>(output, sizes, offsets, output_size);
  } else if (offsets.scalar_type() == at::kLong) {
    expand_csr_cpu_impl<int64_t>(output, sizes, offsets, output_size);
  } else {
    AT_ERROR("Unsupported type (expand_csr_cuda)");
  }

  return {output, sizes};
}

TORCH_LIBRARY_IMPL(epic_ops, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::expand_csr"),
         TORCH_FN(expand_csr_cpu));
}

} // namespace epic_ops::expand
