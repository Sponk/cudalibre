#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

struct Vector
{
	float* data;
	unsigned int size;
};

__global__ void add(struct Vector a, struct Vector b, struct Vector c)
{
	unsigned int id = threadIdx.x;
	c.data[id] = a.data[id] + b.data[id];
}

const unsigned int TESTSIZE = 16;
//#define TESTSIZE 16
int main(int argc, char* argv[])
{

	Vector dav, dbv, dcv;
	float *a, *b, *c;

	size_t pitch;

	CUDA_CHECK(cudaMalloc((void**) &dav.data, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &dbv.data, sizeof(float) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &dcv.data, sizeof(float) * TESTSIZE));

   	a = new float[TESTSIZE];
   	b = new float[TESTSIZE];
   	c = new float[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++) a[i] = i;
	for(int i = 0; i < TESTSIZE; i++) b[i] = TESTSIZE - i;

	CUDA_CHECK(cudaMemcpy(dav.data, a, sizeof(float) * TESTSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dbv.data, b, sizeof(float) * TESTSIZE, cudaMemcpyHostToDevice));

	dav.size = TESTSIZE;

	// Test addition kernel
	add<<<1, 16>>>(dav, dbv, dcv);
	CUDA_CHECK_LAST;

	CUDA_CHECK(cudaMemcpy2D(dcv.data, pitch, c, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyDeviceToHost));

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
		if(fabs(c[i] - (a[i] + b[i])) > 0.0001)
		{
			std::cerr << "Error: Result is not as expected!" << std::endl;
			retval = 1;
		}

	CUDA_CHECK(cudaFree(dav.data));
	CUDA_CHECK(cudaFree(dbv.data));
	CUDA_CHECK(cudaFree(dcv.data));

	delete[] a;
	delete[] b;
	delete[] c;

	return retval;
}
