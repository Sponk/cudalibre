#include <stdio.h>

__global__ void testkernel(int j, float f)
{
	printf("HI");
}

int main(int argc, char* argv[])
{
	testkernel<<<32, 32>>>(12, 32.00f);
}
