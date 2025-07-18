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
    return (x >> 4) & 0x3FFFF;
}

__device__ static inline uint64_t make_smem_desc(bf16* ptr, int ld)
{
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t description = 0ull;

    description |= matrix_descriptor_encode(address);
    description |= matrix_descriptor_encode(ld) << 16;
    description |= matrix_descriptor_encode(16) << 32;
    description |= 1ull << 62;

    return description;
}
