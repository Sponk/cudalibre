#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

__global__ void addAbs(float* a, float* b, float* c)
{
	unsigned int id = threadIdx.x;
	c[id] = fabs(a[id] + b[id]);
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	float *da, *db, *dc, *a, *b, *c;

	size_t pitch;

	CUDA_CHECK(cudaMalloc((void**) &da, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &db, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &dc, sizeof(float) * TESTSIZE));

   	a = new float[TESTSIZE];
   	b = new float[TESTSIZE];
   	c = new float[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++) a[i] = i;
	for(int i = 0; i < TESTSIZE; i++) b[i] = TESTSIZE - i;

	CUDA_CHECK(cudaMemcpy(da, a, sizeof(float) * TESTSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(db, b, sizeof(float) * TESTSIZE, cudaMemcpyHostToDevice));

	// Test addition kernel
	addAbs<<<1, 32>>>(da, db, dc);
	CUDA_CHECK_LAST;

	CUDA_CHECK(cudaMemcpy2D(dc, pitch, c, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyDeviceToHost));

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
		if(fabs(c[i] - fabs(a[i] + b[i])) > 0.0001)
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
