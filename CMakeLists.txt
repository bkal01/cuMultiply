cmake_minimum_required(VERSION 3.10)
project(cuMultiply LANGUAGES CXX CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/kernels)

# Collect all kernel source files
file(GLOB KERNEL_SOURCES "kernels/*_multiply.cu")

# Create a library for each kernel
foreach(KERNEL_SOURCE ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SOURCE} NAME_WE)
    add_library(${KERNEL_NAME} SHARED ${KERNEL_SOURCE})
    set_target_properties(${KERNEL_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endforeach()

# Add test kernels library
add_library(device_helpers_test SHARED
    kernels/device_helpers_test.cu
)
set_target_properties(device_helpers_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Installation
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION lib
        FILES_MATCHING PATTERN "*.so") 