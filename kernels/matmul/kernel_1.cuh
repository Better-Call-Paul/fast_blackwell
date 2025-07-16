#pragma once

#include "../cuda_common.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
using bf16 = __nv_bfloat16;

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

__device__ inline void tcgen05_mma_f16_group1(
    unsigned long long d_ptr,
    unsigned long long a_desc,
    unsigned long long b_desc,
    unsigned int        idesc,
    unsigned int        disable_output_lane,
    unsigned int        enable_input_d
)
{
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%4}, %5;\n"
        :
        : "l"(d_ptr),
          "l"(a_desc),
          "l"(b_desc),
          "r"(idesc),
          "r"(disable_output_lane),
          "r"(enable_input_d)
    );
}

__device__ inline void tcgen05_mma_f16_group2(
    unsigned long long d_ptr,
    unsigned long long a_desc,
    unsigned long long b_desc,
    unsigned int        idesc,
    unsigned int        disable_output_lane,
    unsigned int        enable_input_d
)
{
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%4}, %5;\n"
        :
        : "l"(d_ptr),
          "l"(a_desc),
          "l"(b_desc),
          "r"(idesc),
          "r"(disable_output_lane),
          "r"(enable_input_d)
    );
}

extern __shared__ bf16 smem[];

__global__ void kernel_group1(
const bf16* A,
const bf16* B,
float* C,
int M,
int N,
int K
)
{
    /*
     Valid Locations:
     A: TMEM or SMEM
     B: SMEM in current CTA 
     Accumulator C: TMEM
     */
    bf16* smem_A = smem;
    bf16* smem_B = smem + M * K;
    bf16* smem_D = smem + M * K + K * N;
    int lane_id  = threadIdx.x & 31;
    int a_size   = M * K;
    int b_size   = K * N;
    int d_size   = M * N;

    for (int i = lane_id; i < a_size; i += 32)
    {
        smem_A[i] = A[i];
    }

    for (int i = lane_id; i < b_size; i += 32)
    {
        smem_B[i] = B[i];
    }

    __syncthreads();

    unsigned long long a_desc = make_smem_desc(smem_A, M);
    unsigned long long b_desc = make_smem_desc(smem_B, K);
    unsigned long long d_desc = make_smem_desc(smem_D, M);
    unsigned int idesc        = M | (N << 8) | (K << 16);
    unsigned int disable      = 0;
    unsigned int enable       = 0;

    tcgen05_mma_f16_group1(
    d_desc,
    a_desc,
    b_desc,
    idesc,
    disable,
    enable
    );

    __syncthreads();

    for (int i = lane_id; i < d_size; i += 32)
    {
        C[i] = float(smem_D[i]);
    }
}