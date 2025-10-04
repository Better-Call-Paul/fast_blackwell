cmake_minimum_required(VERSION 3.22)
project(Fast_Blackwell LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 100)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -w")
set(CMAKE_CUDA_FLAGS_RELEASE "--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(matmul_bench matmul_bench.cu)

target_include_directories(matmul_bench PRIVATE
  "${CMAKE_SOURCE_DIR}/kernels"
)

target_link_libraries(matmul_bench PRIVATE
  cublas
  cuda
)
