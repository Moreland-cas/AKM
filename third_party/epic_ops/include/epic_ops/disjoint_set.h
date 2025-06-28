#pragma once

#include <cuda_runtime.h>

#include "epic_ops/utils/atomic.h"

namespace epic_ops::disjoint_set {

template <typename index_t, bool compress_path = true>
__host__ __device__
index_t find(index_t u, index_t* parent_ptr) {
  auto cur = parent_ptr[u];
  if (cur != u) {
    auto next = parent_ptr[cur], prev = u;
    while (cur > next) {
      if constexpr (compress_path) {
        parent_ptr[prev] = next;
        prev = cur;
      }
      cur = next;
      next = parent_ptr[cur];
    }
  }
  return cur;
}

template <typename index_t>
__host__ __device__
void merge(index_t u, index_t v, index_t* parent_ptr) {
  while (u != v) {
    if (u > v) {
      // swap
      auto tmp = u;
      u = v;
      v = tmp;
    }
    auto v_next = utils::atomicCAS(parent_ptr + v, v, u);
    if (v_next == v) {
      break;
    }
    v = v_next;
  }
}

} // namespace epic_ops::disjoint_set
