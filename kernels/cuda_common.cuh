#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cuda_texture_types.h>
#include <surface_types.h>
#include <cuda/pipeline>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
using bf16 = __nv_bfloat16;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
        exit(1);
    }
}

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

__device__ static __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) 
{
    return (x & 0x3FFFF) >> 4;
}

__device__ static __forceinline__ uint64_t make_shared_memory_descriptor(void*      ptr,
                                              uint32_t   ld_bytes,
                                              uint32_t   stride_bytes,
                                              bool       ld_address_mode,
                                              uint8_t    base_offset,
                                              uint8_t    swizzle_mode)
{
    uint64_t p    = reinterpret_cast<uint64_t>(ptr);
    uint64_t d0   = matrix_descriptor_encode(p);
    uint64_t d1   = matrix_descriptor_encode(ld_bytes);
    uint64_t d2   = matrix_descriptor_encode(stride_bytes);
    uint64_t desc = 0;
    desc |= d0;
    desc |= d1       << 16;
    desc |= d2       << 32;
    desc |= uint64_t(1)       << 46;
    desc |= uint64_t(base_offset & 0x7) << 49;
    desc |= uint64_t(ld_address_mode)   << 52;
    desc |= uint64_t(0xB0)    << 53;
    desc |= uint64_t(swizzle_mode & 0x7) << 61;
    return desc;
}

__device__ inline uint64_t make_smem_desc(const __nv_bfloat16* ptr, int ld)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t d0   = ((addr >> 4) & 0x3FFFFull);
    uint64_t d1   = (((ld   >> 4) & 0x3FFFFull) << 16);
    uint64_t d2   = (((16   >> 4) & 0x3FFFFull) << 32);
    return d0 | d1 | d2 | (1ull<<62);
}