#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

__global__ void add(float* a, float* b, float* c)
{
	unsigned int id = threadIdx.x;
	c[id] = a[id] + b[id];
}

__global__ void mul(float* a, float* b, float* c)
{
	unsigned int id = threadIdx.x;
	c[id] = a[id] * b[id];
}

__global__ void div(float* a, float* b, float* c)
{
	unsigned int id = threadIdx.x;
	c[id] = a[id] / b[id];
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	float *a, *b, *c;

	size_t pitch;

	CUDA_CHECK(cudaMallocManaged((void**) &a, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMallocManaged((void**) &b, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMallocManaged((void**) &c, sizeof(float) * TESTSIZE));

   	for(int i = 0; i < TESTSIZE; i++) a[i] = i;
	for(int i = 0; i < TESTSIZE; i++) b[i] = TESTSIZE - i;

	// Test addition kernel
	add<<<1, TESTSIZE>>>(a, b, c);
	CUDA_CHECK_LAST;
	CUDA_CHECK(cudaDeviceSynchronize());

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
		if(fabs(c[i] - (a[i] + b[i])) > 0.0001)
		{
			std::cerr << "Error: Result is not as expected!" << std::endl;
			retval = 1;
		}

	CUDA_CHECK(cudaFree(a));
	CUDA_CHECK(cudaFree(b));
	CUDA_CHECK(cudaFree(c));

	return retval;
}