cmake_minimum_required(VERSION 3.17)
project(adding-two-vectors-cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(adding-two-vectors-cuda main.cu)

set_target_properties(
        adding-two-vectors-cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)