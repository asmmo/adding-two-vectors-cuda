#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>

using namespace std::chrono_literals;
// Kernel definition
__global__ void vectorSum( float const *  v1, float const *  v2, float *  v3)
{
    v3[threadIdx.x] = v1[threadIdx.x] + v2[threadIdx.x];
}


int main()
{
    unsigned int count = 50000000;

    std::vector<float> hVec1(count, 2.2f);
    std::vector<float> hVec2(count, 1.1f);
    std::vector<float> hRes(count, 0.0f);
    std::vector<float> cdRes(count, 0.0f);

    auto st = std::chrono::system_clock::now();

    float* dVec1{};
    cudaMalloc(&dVec1, count * sizeof(float));

    float* dVec2{};
    cudaMalloc(&dVec2, count * sizeof(float));

    float* dRes{};
    cudaMalloc(&dRes, count * sizeof(float));


    cudaMemcpy(dVec1, hVec1.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dVec2, hVec2.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks{};
    dim3 threads{count};
    vectorSum<< < blocks, threads >> > (dVec1, dVec2, dRes);
    cudaMemcpy(cdRes.data(), dRes, count*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dVec1);
    cudaFree(dVec2);
    cudaFree(dRes);

    std::cout << " \n CUDA took " << (std::chrono::system_clock::now() - st).count() << " nano seconds ";


    //To display
    //for (auto& el : hRes) std::cout << el << " ";


    st = std::chrono::system_clock::now();
    std::transform(hVec1.begin(), hVec1.end(), hVec2.begin(), hRes.begin(), [](const auto & i, const auto & j) {return i + j; });
    std::cout << " \n normal took " << (std::chrono::system_clock::now() - st).count() << " nano seconds ";

    int indicate{};
    std::cout<<"\n\n\n\n\n";
    for (size_t i{}; i < cdRes.size(); i++) indicate += cdRes[i] - hRes[i];

    std::cout << "\n\n indicator = " << indicate;
    cudaDeviceReset();

}
