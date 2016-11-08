#ifndef CUDALIBRE_CUDADEVICE_H
#define CUDALIBRE_CUDADEVICE_H

#include "cudalibre.h"
#include <string>
#include <unordered_map>

namespace cu
{
class CudaDevice
{
	cl::Context& context;
	cl::Device device;
	cl::Program program;
	cl::CommandQueue queue;

	std::unordered_map<std::string, cl::Kernel> kernels;

	std::string kernelcode;
	bool kernelCompiled = false;

	size_t bufferHeapIndex = 0;
	std::unordered_map<size_t, cl::Buffer> bufferHeap;

public:
	CudaDevice(cl::Context& context, cl::Device& device);
	CudaDevice(cl::Context& context, cl::Device& device, const std::string& kernelcode);
	const cl::Device& getDevice() { return device; }

	cudaError_t deviceSynchronize() { queue.finish(); return cudaSuccess; }
	cudaError_t mallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
	cudaError_t free(void* ptr);
	cudaError_t memcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

	bool hasKernel() { return kernelCompiled; }
	cudaError_t buildKernel(const char* sources);
	cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const cu::ArgumentList& args = cu::ArgumentList());

	void setSources(const char* sources);
};
}

#endif //CUDALIBRE_CUDADEVICE_H
