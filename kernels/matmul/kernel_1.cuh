#pragma once

#include "../cuda_common.cuh"


template<int 

template<uint BM, uint BN, uint BK, uint TM, uint TN>
__global__ void mma_kernel(int M, int N, int K, __nv_bfloat16 *A, __nv_bfloat *B, fl)
{
    // TMEM 
    


}

