#pragma once

#include "../cuda_common.cuh"

using barrier = cuda::barrier<cuda::thread_scope_cta>;
namespace cde = cuda::device::experimental;
typedef __nv_bfloat16 bf16;



template<uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void mma_kernel(int M, int N, int K, __nv_bfloat16 *A, __nv_bfloat *B, fl)
{
    // TMEM 
    


}

