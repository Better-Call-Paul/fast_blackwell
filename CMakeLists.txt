cmake_minimum_required(VERSION 3.30)
project(Fast_Blackwell LANGUAGES CXX CUDA)

if(NOT DEFINED DEV_DIR)
  if(DEFINED ENV{DEV_DIR})
    set(DEV_DIR $ENV{DEV_DIR})
  elseif(EXISTS "/workspace/devlibs")
    set(DEV_DIR "/workspace/devlibs")
  elseif(EXISTS "$ENV{HOME}/devlibs")
    set(DEV_DIR "$ENV{HOME}/devlibs")
  else()
    message(FATAL_ERROR "DEV_DIR not set; either export ENV{DEV_DIR}, pass -DDEV_DIR=path, or ensure /workspace/devlibs or ~/devlibs exists")
  endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES 90)

add_subdirectory(
  "${DEV_DIR}/nvbench"
  "${CMAKE_BINARY_DIR}/nvbench"
  EXCLUDE_FROM_ALL
)

add_executable(matmul_bench matmul_bench.cu)

target_include_directories(matmul_bench PRIVATE
  "${DEV_DIR}/nvbench/include"
  "${CMAKE_SOURCE_DIR}/kernels"
)

target_link_libraries(matmul_bench PRIVATE nvbench::main)
