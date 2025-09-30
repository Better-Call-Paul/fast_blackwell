#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <unistd.h>
#include <cmath>

#include "kernel_runners.cuh"

__global__ void warmupKernel()
{
    __shared__ int s[100];
    s[0] += s[1];
}

std::default_random_engine generator(69);
int yo = 0;
void randomize_matrix(__nv_bfloat16 *mat, int N) 
{
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) 
  {
    mat[i] = distribution(generator);
  }
  ++yo;
}

bool verify_matrix(__nv_bfloat16 *matRef, __nv_bfloat16 *matOut, int N) 
{
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) 
  {
    int r = i / 8192, c = i % 8192;
    int it = c*8192+r;
    diff = std::fabs(__bfloat162float(matRef[i]) - __bfloat162float(matOut[i]));
    if (diff > 0.1) 
    {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

void run_kernel(int kernel_num, int M, int N, int K, __nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int *DB = nullptr)
{
  
  
}

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, __nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C)
{
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) 
  {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}


static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

int main()
{
	cudaError_t cudaStatus;

	// will be fp16 and then fp8 e4m3

}
