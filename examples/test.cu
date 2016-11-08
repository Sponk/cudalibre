#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void testkernel(__device__ float* a, __device__ float* b, __device__ float* c)
{
	unsigned int id = threadIdx.x;
	c[id] = a[id] + b[id];
}

void print_cuda_devices()
{
	int num = 0;
  	struct cudaDeviceProp devprop;

  	cudaGetDeviceCount(&num);
  	if(!num)
    {
    	std::cout << "No CUDA devices found." << std::endl;
      	return;
    }

  	for(int i = 0; i < num; i++)
	{
  		cudaGetDeviceProperties(&devprop, i);
      	std::cout << "Found CUDA device: " << devprop.name << std::endl;
		std::cout << "\tCompute Capability: " << devprop.major << "." << devprop.minor << std::endl;
		std::cout << "\tMultiprocessor Count: " << devprop.multiProcessorCount << std::endl;
		std::cout << "\tClock Rate: " << static_cast<float>(devprop.clockRate)/1024.0f/1024.0f << "Ghz" << std::endl;
		std::cout << "\tTotal Global: " << devprop.totalGlobalMem/1024 << "MB" << std::endl;
		std::cout << "\tTotal L2 Cache: " << devprop.l2CacheSize << "KB" << std::endl;
	}
}

#define TESTSIZE 32
int main(int argc, char* argv[])
{
	print_cuda_devices();

	float *da, *db, *dc, *a, *b, *c;

	size_t pitch;

   	cudaMallocPitch((void**) &da, &pitch, sizeof(float), TESTSIZE);
   	cudaMallocPitch((void**) &db, &pitch, sizeof(float), TESTSIZE);
   	cudaMallocPitch((void**) &dc, &pitch, sizeof(float), TESTSIZE);

   	a = new float[TESTSIZE];
   	b = new float[TESTSIZE];
   	c = new float[TESTSIZE];

   	for(int i = 0; i < TESTSIZE; i++) a[i] = i;
	for(int i = 0; i < TESTSIZE; i++) b[i] = i;

   	cudaMemcpy2D(da, pitch, a, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy2D(db, pitch, b, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyHostToDevice);

	testkernel<<<1, TESTSIZE>>>(da, db, dc);

	cudaMemcpy2D(dc, pitch, c, sizeof(float), sizeof(float), TESTSIZE, cudaMemcpyDeviceToHost);

	for(int i = 0; i < TESTSIZE; i++)
		printf("%f + %f = %f\n", a[i], b[i], c[i]);
}
