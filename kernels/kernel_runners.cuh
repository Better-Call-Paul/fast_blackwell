#pragma once

#include "kernels.cuh"

void run_kernel_1(int M, int N, int K, __nv_bfloat16 *A, __nv_bfloat16 *B, float *C)
{
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * WARP_SIZE;
    int num_warps = CEIL_DIV(M, 16);
    int blocks = CEIL_DIV(num_warps, warps_per_block);
    dim3 grid
    (
        blocks
    );
    dim3 block
    (
        threads_per_block
    );

    kernel_1<<<grid, block>>>(M, N, K, A, B, C);
    cudaCheck(cudaGetLastError());
}