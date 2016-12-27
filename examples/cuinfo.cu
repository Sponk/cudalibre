#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char** argv)
{
	int num = 0;
	struct cudaDeviceProp devprop;

	cudaGetDeviceCount(&num);
	if(!num)
	{
		std::cout << "No CUDA devices found." << std::endl;
		return 1;
	}

	for(int i = 0; i < num; i++)
	{
		cudaGetDeviceProperties(&devprop, i);
		std::cout << "Found CUDA device: " << devprop.name << std::endl;
		std::cout << "\tCompute Capability: " << devprop.major << "." << devprop.minor << std::endl;
		std::cout << "\tMultiprocessor Count: " << devprop.multiProcessorCount << std::endl;
		std::cout << "\tClock Rate: " << static_cast<float>(devprop.clockRate)/1024.0f/1024.0f << std::endl;
		std::cout << "\tTotal Global: " << devprop.totalGlobalMem/1024 << std::endl;
		std::cout << "\tTotal L2 Cache: " << devprop.l2CacheSize << std::endl;
	}
	return 0;
}
