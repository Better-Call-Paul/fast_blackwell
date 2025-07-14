#pragma once

#include "../cuda_common.cuh"

template<int BM, int BN, int BK>
__global__ void __launch_bounds() tmmem_kernel()
{

    /*
     Valid Locations:
     A: TMEM or SMEM
     B: SMEM
     Accumulator: TMEM
     */

}
