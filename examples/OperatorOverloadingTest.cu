#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

inline __host__ __device__ float2 operator+(const float2 a, const float2 b) 
{ 
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator-(const float2 a, const float2 b) 
{ 
    return make_float2(a.x - b.x, a.y - b.y);
}

__global__ void add(float2* a, float2* b, float2* c)
{
	unsigned int id = threadIdx.x;
	c[id] = a[id] + b[id];
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	float2 *da, *db, *dc, *a, *b, *c;

	size_t pitch;

	CUDA_CHECK(cudaMalloc((void**) &da, sizeof(float2) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &db, sizeof(float2) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &dc, sizeof(float2) * TESTSIZE));

   	a = new float2[TESTSIZE];
   	b = new float2[TESTSIZE];
   	c = new float2[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++) a[i] = make_float2(i, i);
	for(int i = 0; i < TESTSIZE; i++) b[i] = make_float2(TESTSIZE - i, i);

	CUDA_CHECK(cudaMemcpy(da, a, sizeof(float2) * TESTSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(db, b, sizeof(float2) * TESTSIZE, cudaMemcpyHostToDevice));

	// Test addition kernel
	add<<<1, 32>>>(da, db, dc);
	CUDA_CHECK_LAST;

	CUDA_CHECK(cudaMemcpy2D(dc, pitch, c, sizeof(float2), sizeof(float2), TESTSIZE, cudaMemcpyDeviceToHost));

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
	{
		float2 value = c[i] - (a[i] + b[i]);
		if(fabs(value.x) > 0.0001 || fabs(value.y) > 0.0001)
		{
			std::cerr << "Error: Result is not as expected!" << std::endl;
			retval = 1;
		}
	}

	CUDA_CHECK(cudaFree(da));
	CUDA_CHECK(cudaFree(db));
	CUDA_CHECK(cudaFree(dc));

	delete[] a;
	delete[] b;
	delete[] c;

	return retval;
}
