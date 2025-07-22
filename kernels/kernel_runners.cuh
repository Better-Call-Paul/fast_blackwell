#pragma once

#include "kernels.cuh"

void run_kernel_1(int M, int N, int K, __nv_bfloat16* A, __nv_bfloat16* B, float* C)
{
    int lda = K;
    int ldb = N;
    int ldc = N;
    dim3 grid
    {
        (N + 7) / 8,
        (M + 15) / 16
    };
    dim3 block
    {
        128
    };
    kernel_1<<<grid, block>>>
    (
        A,
        B,
        C,
        lda,
        ldb,
        ldc
    );
    cudaDeviceSynchronize();
}
