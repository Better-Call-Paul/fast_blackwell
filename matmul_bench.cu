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

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K)
{
	#pragma omp parallel for collapse(2) // speed
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++i)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; ++k)
			{
				sum += a[i * K + k] * b[j * N + k];
			}
			c[i * N + j] = sum;
		}
	}
}

void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block)
{

}

int run_benchmark(size_t M, size_t N, size_t K)
{
	cudaError_t cudaStatus;

	std::cout << "--------------------  M = " << M << " N = " << N << " K = " << K << "  --------------------\n";
	std::cout << "Block size: " << Mb * 2 << " x " << Nb << "\n";

	// Allocate host memory
	float *h_A = new float[M * K];
	float *h_B = new float[K * N];
	float *h_C = new float[M * N];
	float *h_C_ref = new float[M * N];

	std::cout << "Allocated host memory\n";

	// Initialize random number generator
	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_real_distribution<> dis(-0.5, 0.5);

	// Initialize matrices with random values
	for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
	for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

	std::cout << "Initialized matrices\n";
	
	cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

	std::cout << "Performed CPU matrix multiplication\n";

	__nv_bfloat16 *d_A, *d_B, *d_C;
	cudaCheck(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
	cudaCheck(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
	cudaCheck(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));

	std::cout << "Allocated device memory\n";

	// Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device\n";

    dim3 grid(1, 1);
    dim3 block(1, 1);

    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (false ? 0 : 1); i++) { // warmup
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }

    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (true ? 1 : 5);
    for(int i = 0; i < ITERS; i++) {
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // TFlops
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel Execution Time: " << useconds << " us\n";
    std::cout << "Achieved Performance: " << tflops << " TFLOPS\n";

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    float average_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) 
	{
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) 
		{ // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
        average_error += error;
    }
    average_error /= M*N;

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Average error: " << average_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main()
{
	int N;
    // N = 1024;
    // run_benchmark(N, N, N);
    // N = 2048;
    // run_benchmark(N, N, N);
    // N = 4096;
    // run_benchmark(N, N, N);
    N = 8192;
    run_benchmark(N, N, N);
    N = 16384;
    run_benchmark(N, N, N);
    return 0;
}
