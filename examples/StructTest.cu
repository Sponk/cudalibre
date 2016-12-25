#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

typedef struct
{
	float x;
	float y;
	float z;
} vec3;

__global__ void add(vec3* a, vec3* b, vec3* c)
{
	unsigned int id = threadIdx.x;
	c[id].x = a[id].x + b[id].x;
	c[id].y = a[id].y * b[id].y;
	c[id].z = a[id].z / b[id].z;
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	vec3 *da, *db, *dc, *a, *b, *c;

	int pitch;

	CUDA_CHECK(cudaMalloc((void**) &da, sizeof(vec3) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &db, sizeof(vec3) * TESTSIZE));
	CUDA_CHECK(cudaMalloc((void**) &dc, sizeof(vec3) * TESTSIZE));

   	a = new vec3[TESTSIZE];
   	b = new vec3[TESTSIZE];
   	c = new vec3[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++)
   	{
   		a[i].x = i;
   		a[i].y = -i;
   		a[i].z = i + 1;
   	}

	for(int i = 0; i < TESTSIZE; i++)
	{
    	b[i].x = TESTSIZE - i;
       	b[i].y = TESTSIZE + i;
       	b[i].z = TESTSIZE + i;
	}

	CUDA_CHECK(cudaMemcpy(da, a, sizeof(vec3) * TESTSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(db, b, sizeof(vec3) * TESTSIZE, cudaMemcpyHostToDevice));

	// Test addition kernel
	add<<<1, 32>>>(da, db, dc);
	CUDA_CHECK_LAST;

	CUDA_CHECK(cudaMemcpy2D(dc, pitch, c, sizeof(vec3), sizeof(vec3), TESTSIZE, cudaMemcpyDeviceToHost));

	// Check expected result
	int retval = 0;
	for(int i = 0; i < TESTSIZE; i++)
		if(CMP_FLOAT(c[i].x, a[i].x + b[i].x, 0.0001)
		|| CMP_FLOAT(c[i].y, a[i].y * b[i].y, 0.0001)
		|| CMP_FLOAT(c[i].z, a[i].z / b[i].z, 0.0001))
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
