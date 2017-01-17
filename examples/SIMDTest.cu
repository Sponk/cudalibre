#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common.h"

__global__ void add(float2* a, float2* b, float2* c)
{
	unsigned int id = threadIdx.x;
	c[id].x = a[id].x + b[id].x;
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	float2 value;
	return 0;
}
