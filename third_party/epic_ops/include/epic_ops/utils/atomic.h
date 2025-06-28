#pragma once

#include <cuda_runtime.h>

namespace epic_ops::utils {

template <typename dst_t, typename src_t>
__forceinline__ __host__ __device__
dst_t type_reinterpret(src_t value)
{
    return *(reinterpret_cast<dst_t*>(&value));
}

__forceinline__ __device__
int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  auto ret = ::atomicCAS(
      reinterpret_cast<unsigned long long *>(address),
      type_reinterpret<unsigned long long>(compare),
      type_reinterpret<unsigned long long>(val));
  return type_reinterpret<int64_t>(ret);
}

__forceinline__ __device__
int32_t atomicCAS(int32_t* address, int32_t compare, int32_t val) {
  return ::atomicCAS(address, compare, val);
}

__forceinline__ __device__
int64_t atomicAdd(int64_t* address, int64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  auto ret = ::atomicAdd(
      reinterpret_cast<unsigned long long *>(address),
      type_reinterpret<unsigned long long>(val));
  return type_reinterpret<int64_t>(ret);
}

__forceinline__ __device__
int32_t atomicAdd(int32_t* address, int32_t val) {
  return ::atomicAdd(address, val);
}

} // namespace epic_ops::utils
