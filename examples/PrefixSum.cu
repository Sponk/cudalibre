#include <cuda_runtime.h>
#include <iostream>
#include "common.h"

#define LOG(thrd, msg, val) if(threadIdx.x == thrd) printf(msg, val);

__global__ void prefixSum(float* in, float* out, unsigned int size)
{
	if(threadIdx.x >= size)
		return;

	__shared__ float buffer[64];
	buffer[threadIdx.x] = ((threadIdx.x == 0) ? 0 : in[threadIdx.x - 1]);
	buffer[threadIdx.x + size] = 0;

	__syncthreads();

	int usingFront = 0;
	int usingBack = 1;
	for(int stride = 1; stride < size; stride *= 2)
	{
		usingFront = 1 - usingFront;
		usingBack = 1 - usingFront;
		const int bufferIdx = threadIdx.x + size*usingFront;
		const int backIdx = threadIdx.x + size*usingBack;

		if(threadIdx.x >= stride)
			buffer[bufferIdx] = buffer[backIdx] + buffer[backIdx - stride];
		else
			buffer[bufferIdx] = buffer[backIdx];

		__syncthreads();
	}

	out[threadIdx.x] = buffer[threadIdx.x + size*usingFront];
}

__global__ void prefixSumInPlace(float* in, unsigned int size)
{
	if(threadIdx.x >= size)
		return;

	__shared__ float buffer[64];
	buffer[threadIdx.x] = ((threadIdx.x == 0) ? 0 : in[threadIdx.x - 1]);
	buffer[threadIdx.x + size] = 0;

	__syncthreads();

	int stride;
	for(stride = 1; stride < size; stride *= 2)
	{
		if(threadIdx.x % stride == 0)
			buffer[threadIdx.x] += buffer[threadIdx.x - stride];

		__syncthreads();
	}

	for(; stride > 0; stride /= 2)
	{
		if(threadIdx.x % stride != 0)
			buffer[threadIdx.x] += buffer[threadIdx.x - stride];

		__syncthreads();
	}

	in[threadIdx.x] = buffer[threadIdx.x];
}

int main(int argc, char** argv)
{
	const unsigned int size = 8;
	float *din, *dout;

	float array[] = {3,1,7,0,4,1,6,3};
	CUDA_CHECK(cudaMalloc((void**) &din, sizeof(float) * size));
	CUDA_CHECK(cudaMalloc((void**) &dout, sizeof(float) * size));

	CUDA_CHECK(cudaMemcpy(din, array, sizeof(float) * size, cudaMemcpyHostToDevice));
	prefixSum<<<1, size>>>(din, dout, size);

	float result[size];
	cudaMemcpy(result, dout, sizeof(float) * size, cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());

	for(int i = 0; i < size; i++)
	{
		std::cout << "SUM: " << array[i] << " -> " << result[i] << std::endl;
	}

	CUDA_CHECK(cudaMemcpy(din, array, sizeof(float) * size, cudaMemcpyHostToDevice));
	prefixSumInPlace<<<1, size>>>(din, size);

	std::cout << "In place: " << std::endl;
	cudaMemcpy(result, din, sizeof(float) * size, cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());

	for(int i = 0; i < size; i++)
	{
		std::cout << "SUM: " << array[i] << " -> " << result[i] << std::endl;
	}

	cudaFree(din);
	cudaFree(dout);
}