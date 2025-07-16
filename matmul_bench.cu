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

#include "cuda_common.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
        exit(1);
    }
}

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

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

int main()
{
  warmupKernel<<<1024, 1024>>>();

  cublasCreate(&cublas_handle);
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int max_size = 8192;
  int M, N, K;
  M = N = K = max_size;

  __nv_bfloat16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
  __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;
  int *device_buffer = nullptr, *d_device_buffer = nullptr;

  A     = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * M * K);
  B     = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * K * N);
  C     = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * M * N);
  C_ref = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * M * N);

  device_buffer = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void**)&d_device_buffer, sizeof(int) * max_size * 128));

  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  randomize_matrix(C, M * N);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(__nv_bfloat16) * M * K));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(__nv_bfloat16) * K * N));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(__nv_bfloat16) * M * N));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(__nv_bfloat16) * M * N));
  
  cudaCheck(cudaMemcpy(dA, A, sizeof(__nv_bfloat16) * M * K, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(__nv_bfloat16) * K * N, cudaMemcpyHostToDevice));

  int repeat_count = 5;
  bool run_verification = false;

  for ( int kernel_num : {0, 1, 2, 3, 4, 5})
  {
    sleep(5);
    std::cout << "KERNEL: " << kernel_num << "\n";

    if (run_verification)
    {
      memset(C, 0, sizeof(__nv_bfloat16) * M * N);
      cudaCheck(cudaMemcpy(dC, C, sizeof(__nv_bfloat16) * M * N, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(__nv_bfloat16) * M * N, cudaMemcpyHostToDevice));
      
      memset(device_buffer, ~0, sizeof(int) * max_size * 128);
      cudaCheck(cudaMemcpy(d_device_buffer, device_buffer, sizeof(int) *max_size, cudaMemcpyHostToDevice));

      runCublasGemmBF16(M, N, K, dA, dB, dC_ref);

      // run_kernel(kernel_num, M, N, K, dA, dB, dC);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
      cudaMemcpy(C, dC, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost);

      if (kernel_num > 1 && !verify_matrix(C_ref, C, M * N))
      {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~\n";
        printf("%f\n", __bfloat162float(C_ref[M]));
      }

      // Retrieve Device Buffer Datapoints

      cudaMemcpy(device_buffer, d_device_buffer, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;

      while (device_buffer[i] != 0)
      {
        sumLoad += device_buffer[i], cntLoad += device_buffer[i + 1];
        sumCompute += device_buffer[i + 2], cntCompute += device_buffer[i + 3];
        sumStore += device_buffer[i + 4], cntStore += device_buffer[i + 5];
        i += 6;
        times++;
      }

      if (times > 0)
      {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n", (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute, (sumStore + .0) / cntStore, times);
      }
    
    }

    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    // dummy launch
    runCublasGemmBF16(M, N, K, dA, dB, dC_ref);

    cublasSetStream(cublas_handle, 0);
    cudaEventRecord(start, 0);

    for (int j = 0; j < repeat_count; ++j)
    {
        runCublasGemmBF16(M, N, K, dA, dB, dC_ref);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long FLOPS = (2LL * M) * (N * K);

    printf(
        "Average elapsed time: (%7.6f) s, performance: (%9.3f) PFLOPS. size: (%ld).\n\n",
        elapsed_time * 1e-3 / repeat_count,
        (repeat_count * FLOPS * 1e-12) / elapsed_time,
        M
    );
        
  }

  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);

  return 0;
}
