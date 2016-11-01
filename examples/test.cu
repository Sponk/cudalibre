#include <stdio.h>

__global__ void testkernel(int j, float f)
{
	printf("gridDim: %f %f %f blockDim: %f %f %f "
			"blockIdx: %f %f %f threadIdx: %f %f %f\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			blockIdx.x, blockIdx.y, blockIdx.z,
			threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char* argv[])
{
	testkernel<<<4, 4>>>(12, 32.00f);
}
