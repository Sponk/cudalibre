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
	float *da, *db, *dc, *a, *b, *c;

	size_t pitch;

   	CUDA_CHECK(cudaMallocPitch((void**) &da, &pitch, sizeof(float), TESTSIZE));
   	CUDA_CHECK(cudaMallocPitch((void**) &db, &pitch, sizeof(float), TESTSIZE));
   	CUDA_CHECK(cudaMallocPitch((void**) &dc, &pitch, sizeof(float), TESTSIZE));

   	a = new float[TESTSIZE];
   	b = new float[TESTSIZE];
   	c = new float[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++) a[i] = i;
	for(int i = 0; i < TESTSIZE; i++) b[i] = TESTSIZE - i;

   	CUDA_CHECK(cudaMemcpy2D(da, pitch, a, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(db, pitch, b, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyHostToDevice));

	// Test addition kernel
	add<<<1, 32>>>(da, db, dc);
	CUDA_CHECK_LAST;

	CUDA_CHECK(cudaMemcpy2D(dc, pitch, c, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyDeviceToHost));

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
		if(fabs(c[i] - (a[i] + b[i])) > 0.0001)
		{
			std::cerr << "Error: Result is not as expected!" << std::endl;
			retval = 1;
		}

	CUDA_CHECK(cudaFree(da));
	CUDA_CHECK(cudaFree(db));
	CUDA_CHECK(cudaFree(dc));

	delete[] a;
	delete[] b;
	delete[] c;

	return retval;
}