#ifndef CUDALIBRE_CUDADEVICE_H
#define CUDALIBRE_CUDADEVICE_H

#include "cudalibre.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace cu
{
/**
 * @brief Represents a CUDA device.
 */
class CudaDevice
{
	cl::Context& context;
	cl::Device device;
	cl::Program program;
	cl::CommandQueue queue;

	std::unordered_map<std::string, cl::Kernel> kernels;

	std::string kernelcode;
	bool kernelCompiled = false;

	struct UnifiedBuffer
	{
		UnifiedBuffer() {}
		UnifiedBuffer(void* host, cl::Buffer* buffer) :
			host(host),
			buffer(buffer) {}

		void* host = nullptr;
		cl::Buffer* buffer;
	};

	//size_t bufferHeapIndex = 0;
	std::unordered_map<void*, UnifiedBuffer> bufferHeap;

public:
	CudaDevice(cl::Context& context, cl::Device& device);
	CudaDevice(cl::Context& context, cl::Device& device, const std::string& kernelcode);

	/**
	 * @brief Fetches the OpenCL device object.
	 * @return The OpenCL device.
	 */
	const cl::Device& getDevice() { return device; }

	/**
	 * @see cudaDeviceSynchronize
	 */
	cudaError_t deviceSynchronize() { queue.finish(); return cudaSuccess; }

	/**
	 * @see cudaMallocPitch
	 */
	cudaError_t mallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
	cudaError_t mallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);

	/**
	 * @see cudaFree
	 */
	cudaError_t free(void* ptr);

	/**
	 * @see cudaMemcpy2D
	 */
	cudaError_t memcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

	cudaError_t malloc(void** devPtr, size_t size);
	cudaError_t memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

	/**
	 * @brief Indicates if the kernel has been built already.
	 * @return True if it has, false if it has not.
	 */
	bool hasKernel() { return kernelCompiled; }

	/**
	 * @brief Builds the registered kernel code.
	 * @param sources The OpenCL sources.
	 * @return A CUDA error code or cudaSuccess.
	 */
	cudaError_t buildKernel(const char* sources);

	/**
	 * @brief Calls a kernel on the device.
	 * @param name The kernel name.
	 * @param gridsize The CUDA gridsize.
	 * @param blocksize The CUDA blocksize.
	 * @param args A list of arguments.
	 * @return cudaSuccess or an error code.
	 */
	cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const cu::ArgumentList& args = cu::ArgumentList());

	void setSources(const char* sources);
	void getProgramBinaries(std::vector<unsigned char>& data);

	void clear()
	{
		kernels.clear();
		bufferHeap.clear();
	}
};
}

#endif //CUDALIBRE_CUDADEVICE_H
