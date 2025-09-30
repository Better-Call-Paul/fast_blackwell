#pragma once

#include "../cuda_common.cuh"

constexpr int WARP_SIZE = 32;

__global__ void kernel_1(const __nv_bfloat16* A,
                                  const __nv_bfloat16* B,
                                  float*                C,
                                  int                   lda,
                                  int                   ldb,
                                  int                   ldc)
{

}

__global__ void basic_tcgen05_mma_kernel()
{
    
}
