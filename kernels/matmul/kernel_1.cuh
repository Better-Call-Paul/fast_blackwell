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
    __shared__ __nv_bfloat16 As[16][16];
    __shared__ __nv_bfloat16 Bs[16][ 8];
    int tx = threadIdx.x;
    int row = blockIdx.y * 16 + tx / 16;
    int col = blockIdx.x *  8 + tx %  8;
    As[tx/16][tx%16] = A[row*lda + tx%16];
    Bs[tx/8 ][tx%8 ] = B[(tx/8)*ldb + col];
    __syncthreads();
    uint64_t adesc = make_smem_desc(&As[0][0], 16);
    uint64_t bdesc = make_smem_desc(&Bs[0][0],  8);
    uint32_t idesc = 0x01020490;
    float2 d0;
    d0.x = 0.0f;
    d0.y = 0.0f;
    float2 d1;
    d1.x = 0.0f;
    d1.y = 0.0f;
    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, "
        "[%4], [%5], [%6];\n"
        : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
        : "l"(adesc), "l"(bdesc), "l"(idesc)
    );
    __syncthreads();
    if
    (
        tx == 0
    )
    {
        C[row*ldc + col] = d0.x;
    }
}

__global__ void basic_tcgen05_mma_kernel()
{
    float2 d0;
    float2 d1;
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
    uint32_t a3;
    uint32_t b0;
    uint32_t b1;

    d0.x = 0.0f;
    d0.y = 0.0f;
    d1.x = 0.0f;
    d1.y = 0.0f;

    a0 = 0;
    a1 = 0;
    a2 = 0;
    a3 = 0;
    b0 = 0;
    b1 = 0;

    asm volatile
    (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},"
    "{%4,%5,%6,%7},"
    "{%8,%9},"
    "{%0,%1,%2,%3};"
    : "+f"(d0.x)
    , "+f"(d0.y)
    , "+f"(d1.x)
    , "+f"(d1.y)
    : "r"(a0)
    , "r"(a1)
    , "r"(a2)
    , "r"(a3)
    , "r"(b0)
    , "r"(b1)
    );
}
