 cmake_minimum_required(VERSION 3.10)
project(cuda-multiply-bench LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/benchmark.cpp
    src/kernel_loader.cpp
    src/utils.cpp
)

# Create the executable
add_executable(cuda_bench ${SOURCE_FILES})
target_link_libraries(cuda_bench ${CUDA_LIBRARIES})