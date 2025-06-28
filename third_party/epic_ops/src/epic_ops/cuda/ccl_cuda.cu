#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "epic_ops/ccl.h"
#include "epic_ops/disjoint_set.h"
#include "epic_ops/utils/atomic.h"
#include "epic_ops/utils/thrust_allocator.h"

namespace epic_ops::ccl {

namespace {
  template <typename index_t, bool compacted>
  __host__ __device__
  inline std::tuple<index_t, index_t> get_edge_range(
      const index_t u, const index_t* const __restrict__ indices_ptr) {
    if constexpr (compacted) {
      return {indices_ptr[u], indices_ptr[u + 1]};
    } else {
      return {indices_ptr[u * 2], indices_ptr[u * 2 + 1]};
    }
  }

  template <typename index_t, bool compacted, typename policy_t>
  void initialize(
      const policy_t& policy,
      index_t num_nodes,
      const index_t* const __restrict__ indices_ptr,
      const index_t* const __restrict__ edges_ptr,
      index_t* __restrict__ labels_ptr) {
    // initialize with first smaller neighbor ID
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          const auto [begin, end] = get_edge_range<index_t, compacted>(u, indices_ptr);
          auto parent = u;
          for (auto i = begin; i < end; i++) {
            const auto v = edges_ptr[i];
            if (v < parent) {
              parent = v;
              break;
            }
          }
          labels_ptr[u] = parent;
        });
  }

  template <typename index_t, bool compacted, typename policy_t>
  void merge_1(
      const policy_t& policy,
      index_t num_nodes,
      const index_t* const __restrict__ indices_ptr,
      const index_t* const __restrict__ edges_ptr,
      index_t* __restrict__ labels_ptr,
      index_t* __restrict__ queue_ptr,
      index_t* __restrict__ queue_params_ptr) {
    // merge low-degree nodes at thread granularity and fill work queue
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          if (labels_ptr[u] == u) {
            // 0 degree
            return;
          }

          const auto [begin, end] = get_edge_range<index_t, compacted>(u, indices_ptr);
          const auto degree = end - begin;
          if (degree > 32) {
            index_t idx;
            if (degree <= 352) {
              idx = utils::atomicAdd(queue_params_ptr + 0, 1);
            } else {
              idx = utils::atomicAdd(queue_params_ptr + 1, -1);
            }
            queue_ptr[idx] = u;
          } else {
            auto u_label = disjoint_set::find(u, labels_ptr);
            for (auto i = begin; i < end; i++) {
              const auto v = edges_ptr[i];
              if (u <= v) {
                continue;
              }

              const auto v_label = disjoint_set::find(v, labels_ptr);
              disjoint_set::merge(u_label, v_label, labels_ptr);
            }
          }
        });
  }

  template <typename index_t, bool compacted, int pack_size, bool reverse_queue, typename policy_t>
  void merge_2(
      const policy_t& policy,
      index_t num_nodes,
      index_t num_nodes_enqueued,
      const index_t* const __restrict__ queue_ptr,
      const index_t* const __restrict__ indices_ptr,
      const index_t* const __restrict__ edges_ptr,
      index_t* __restrict__ labels_ptr) {
    // merge medium/large-degree nodes at warp/thread granularity
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes_enqueued * pack_size),
        [=] __host__ __device__ (index_t idx) {
          const index_t lane_idx = idx % pack_size;
          idx /= pack_size;
          const index_t u = reverse_queue ? queue_ptr[num_nodes - 1 - idx] : queue_ptr[idx];

          if (labels_ptr[u] == u) {
            // 0 degree
            return;
          }

          const auto [begin, end] = get_edge_range<index_t, compacted>(u, indices_ptr);
          if (begin + lane_idx >= end) {
            return;
          }

          auto u_label = disjoint_set::find(u, labels_ptr);
          for (auto i = begin + lane_idx; i < end; i += pack_size) {
            const auto v = edges_ptr[i];
            if (u <= v) {
              continue;
            }

            const auto v_label = disjoint_set::find(v, labels_ptr);
            disjoint_set::merge(u_label, v_label, labels_ptr);
          }
        });
}

  template <typename index_t, typename policy_t>
  void flatten(
      const policy_t& policy,
      index_t num_nodes,
      index_t* __restrict__ labels_ptr) {
    // flatten
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          const auto parent = disjoint_set::find<index_t, false>(u, labels_ptr);
          if (parent != u) {
            labels_ptr[u] = parent;
          }
        });
  }
}

template <typename index_t, bool compacted>
void connected_components_labeling_cuda_impl(
    // outputs
    at::Tensor &labels,
    // inputs
    const at::Tensor& indices,
    const at::Tensor& edges) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  index_t num_nodes = labels.size(0);
  index_t num_edges = edges.size(0);

  auto labels_ptr = labels.data_ptr<index_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto edges_ptr = edges.data_ptr<index_t>();

  auto queue = at::empty_like(labels);
  auto queue_params = at::tensor(
      std::vector<index_t>({0, num_nodes - 1})).to(queue.device());

  auto queue_ptr = queue.template data_ptr<index_t>();
  auto queue_params_ptr = queue_params.template data_ptr<index_t>();

  initialize<index_t, compacted>(
      policy, num_nodes, indices_ptr, edges_ptr, labels_ptr);
  merge_1<index_t, compacted>(
      policy, num_nodes, indices_ptr, edges_ptr, labels_ptr, queue_ptr, queue_params_ptr);

  auto queue_params_cpu = queue_params.cpu();
  auto queue_params_cpu_ptr = queue_params_cpu.template data_ptr<index_t>();
  auto num_medium_nodes = queue_params_cpu_ptr[0];
  auto num_large_nodes = num_nodes - queue_params_cpu_ptr[1] - 1;
  if (num_medium_nodes > 0) {
    merge_2<index_t, compacted, 32, false>(
        policy, num_nodes, num_medium_nodes, queue_ptr,
        indices_ptr, edges_ptr, labels_ptr);
  }
  if (num_large_nodes > 0) {
    merge_2<index_t, compacted, 256, true>(
        policy, num_nodes, num_large_nodes, queue_ptr,
        indices_ptr, edges_ptr, labels_ptr);
  }

  flatten<index_t>(policy, num_nodes, labels_ptr);
}

at::Tensor connected_components_labeling_cuda(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted) {
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");

  TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
  TORCH_CHECK(edges.dim() == 1, "edges must be a 1D tensor");

  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(edges.is_contiguous(), "edges must be contiguous");

  auto num_nodes = compacted ? indices.size(0) - 1 : indices.size(0) / 2;
  auto labels = at::empty({num_nodes}, indices.options());

  if (indices.scalar_type() == at::kInt) {
    if (compacted) {
      connected_components_labeling_cuda_impl<int32_t, true>(labels, indices, edges);
    } else {
      connected_components_labeling_cuda_impl<int32_t, false>(labels, indices, edges);
    }
  } else if (indices.scalar_type() == at::kLong) {
    if (compacted) {
      connected_components_labeling_cuda_impl<int64_t, true>(labels, indices, edges);
    } else {
      connected_components_labeling_cuda_impl<int64_t, false>(labels, indices, edges);
    }
  } else {
    AT_ERROR("Unsupported type (connected_components_labeling_cuda)");
  }

  return labels;
}

TORCH_LIBRARY_IMPL(epic_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("epic_ops::connected_components_labeling"),
         TORCH_FN(connected_components_labeling_cuda));
}

} // namespace epic_ops::ccl
