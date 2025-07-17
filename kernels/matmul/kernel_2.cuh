#pragma once

#include "../cuda_common.cuh"

__device__ static __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) 
{
    return (x >> 4) & 0x3FFFF;
}

__device__ static inline uint64_t make_smem_desc(bf16* ptr, int ld)
{
    uint64_t addr_cell = matrix_descriptor_encode((uint64_t)ptr);
    uint64_t ld_cell   = matrix_descriptor_encode((uint64_t)(ld * sizeof(bf16)));
    return
    addr_cell
    | (ld_cell << 16)
    | (ld_cell << 32)
    | (1ULL << 62);
}

/*
template<int BM, int BN, int BK>
__global__ void __launch_bounds() tmmem_kernel()
{

   

}
*/