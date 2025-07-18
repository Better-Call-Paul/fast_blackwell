#pragma once

#include "../cuda_common.cuh"

constexpr int WARP_SIZE = 32;

__global__ void kernel_1(int M, int N, int K, const bf16 *A, const bf16 *B, float *C)
{
    int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int rowStart = globalWarpId * 16;
    if (rowStart >= M)
    {
        return;
    }

    A += rowStart * K;
    C += rowStart * N;

    unsigned c0 = 0u;
    unsigned c1 = 0u;
    unsigned c2 = 0u;
    unsigned c3 = 0u;

    for (int k = 0; k < K; k += WARP_SIZE)
    {
        // load fragments
        unsigned a0 = ((unsigned*)A)[0 * 16 + (threadIdx.x & 31)];
        unsigned a1 = ((unsigned*)A)[1 * 16 + (threadIdx.x & 31)];
        unsigned a2 = ((unsigned*)A)[2 * 16 + (threadIdx.x & 31)];
        unsigned a3 = ((unsigned*)A)[3 * 16 + (threadIdx.x & 31)];

        unsigned b0 = ((unsigned*)(B + k * N))[threadIdx.x & 31];
        unsigned b1 = ((unsigned*)(B + k * N + 16))[threadIdx.x & 31];

        asm volatile 
        (
            "tcgen05.mma.cta_group.kind::f16.cta_group::1 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1)
        );

        A += WARP_SIZE;
    }

    int lane = threadIdx.x & 31;

    C[lane        ] = __int_as_float(c0);
    C[N + lane    ] = __int_as_float(c1);
    C[2*N + lane  ] = __int_as_float(c2);
    C[3*N + lane  ] = __int_as_float(c3);
}
