#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

__global__ void countThreads(int* output)
{
	__shared__ int counter;
	counter = 0;

	__syncthreads();
	atomicInc(&counter, 0);
	__syncthreads();

	if (threadIdx.x == 0)
		*output = counter;
}

int main(int argc, char* argv[])
{
	int* output;
	cudaMallocManaged((void**) &output, 4);
	*output = 0;
	countThreads<<<1, 32>>>(output);
	cudaDeviceSynchronize();

	if(*output != 32)
	{
		std::cerr << *output << std::endl;
		cudaFree(output);
		return 1;
	}

	cudaFree(output);
	return 0;
}
