#include <stdio.h>

__global__ void testkernel(int j, float f)
{
	printf("Test: %d %f\n", j, f);
}

int main(int argc, char* argv[])
{
	testkernel<<<512, 32>>>(12, 32.00f);
}
