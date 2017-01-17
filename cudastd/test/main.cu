#include <iostream>
#include <cuda_runtime.h>

__global__ void testMath()
{
	const float f = 23;
	cospif(f);
}

int main(int argc, char** argv)
{
	testMath<<<1, 1>>>();
}
