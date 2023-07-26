#include "Utils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel(float4* device_memory, const unsigned width, const unsigned height)
{
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > width || y > height)
		return;

	device_memory[y * width + x] = make_float4(static_cast<float>(x) / static_cast<float>(width), static_cast<float>(y) / static_cast<float>(height), 1.0f, 1.0f);
}

void launch_kernel(float4* device_memory, const int width, const int height)
{
	dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

	kernel<<<grid_dim, block_dim>>>(device_memory, width, height);

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());
}