#pragma once

#include "cuda_common.cuh"

template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg = false>
__device__ static inline uint32_t instruction_descriptor()
{
    uint32_t desc = 0;

    if constexpr (sizeof(AB) == 2)
    {
        // either accumulate to float, or the input is half and the output is half
        static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, half>) {
            desc |= 0b000 << 7;  // 16-bit A input type as FP16
            desc |= 0b000 << 10; // 16-bit B input type as FP16
        } else if constexpr (std::is_same_v<AB, bf16>) {
            desc |= 0b001 << 7;  // 16-bit A input type as BF16
            desc |= 0b001 << 10; // 16-bit B input type as BF16
        } else if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        }
        /* fp6 and fp4
        else if constexpr (std::is_same_v<AB, fp6e2m3>) {
            desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
            desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
        }
        else if constexpr (std::is_same_v<AB, fp4e2m3>) {
            desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
            desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
        }
        else if constexpr (std::is_same_v<AB, fp4e3m1>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
        }
        */
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
        
    }
    else 
    {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet");
    }

    return desc;
}

template<typename T_AB, int acc, int ncta = 1>
__device__ static inline void a_tensor_b_shmem(uint32_t d_tt_addr, uint32_t a_tt_addr, uint64_t b_desc, uint32_t i_desc)
{
    if constexpr (std::is_same_v<T_AB, fp8e4m3> || std::is_same_v<T_AB, fp8e5m2>)
    {
        // Not implemented yet
    }
    else 
    {
        if (ncta == 1) // single CTA 
        {
            asm volatile(
                "{.reg .pred p:\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;}\n"
                :: "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(i_desc), "n"(acc)
            );
        }
        else // multi CTA 
        {
            asm volatile(
                "{.reg .pred p:\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;}\n"
                :: "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(i_desc), "n"(acc)
            );
        }
    }
}
