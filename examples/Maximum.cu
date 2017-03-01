#include <cuda_runtime.h>
#include <iostream>
#include "common.h"

#define LOG(thrd, msg, val) if(threadIdx.x == thrd) printf(msg, val);
#define BLOCKSIZE 32

__global__ void maximum(int* in, int* out, unsigned int size)
{
	const int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	__shared__ int tmp;

	if(id >= size)
		return;

	if(id == 0)
	{
		*out = 0;
	}

	if(threadIdx.x == 0)
		tmp = 0;
	
	__syncthreads();
	atomicMax(&tmp, in[id]);
	__syncthreads();
	
	if(threadIdx.x == 0)
		atomicMax(out, tmp);
}

int main(int argc, char** argv)
{
	const unsigned int size = 64;
	int *din, *dout;

	int* array = new int[size]; //[] = {3,1,7,0,4,1,6,3};
	srand(time(0) * time(0));

	for(int i = 0; i < size; i++)
		array[i] = i; //rand() % size * 2;

	CUDA_CHECK(cudaMalloc((void**) &din, sizeof(int) * size));
	CUDA_CHECK(cudaMalloc((void**) &dout, sizeof(int)));

	CUDA_CHECK(cudaMemcpy(din, array, sizeof(int) * size, cudaMemcpyHostToDevice));

	const int blocks = size / BLOCKSIZE;
	const int threads = BLOCKSIZE; //size % BLOCKSIZE;
	maximum<<<blocks, threads>>>(din, dout, size);

	int result = 0;
	CUDA_CHECK(cudaMemcpy(&result, dout, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	std::cout << "{ ";
	for(int i = 0; i < size; i++)
	{
		std::cout << array[i] << " ";
	}
	std::cout << " }" << std::endl;

	std::cout << "Maximum is: " << result << std::endl;

	cudaFree(din);
	cudaFree(dout);

	delete[] array;
}
