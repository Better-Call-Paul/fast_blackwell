cmake_minimum_required(VERSION 3.22)
project(Fast_Blackwell LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 90)

add_executable(matmul_bench matmul_bench.cu)

target_include_directories(matmul_bench PRIVATE
  "${CMAKE_SOURCE_DIR}/kernels"
)

target_link_libraries(matmul_bench PRIVATE cublas)
