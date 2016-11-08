#ifndef __CUDALIBRE_H__
#define __CUDALIBRE_H__

#include <stddef.h>
#include <utility> // So std::pair can be used for kernel calls
#include <vector>
#include <memory>
#include <cstring>

#define CL_HPP_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else

#if !defined(USE_CL1) && !defined(WIN32) && !defined(__CYGWIN__)
#include <CL/cl2.hpp>
#else
#include <CL/cl.hpp>
#endif // USE_CL1
#endif

// TODO: https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html
//		/group__CUDART__TYPES_g3f51e3575c2178246db0a94a430e0038.html#g3f51e3575c2178246db0a94a430e0038
typedef enum cudaError
{
	cudaSuccess,
	cudaErrorMemoryAllocation,
	cudaErrorInitializationError,
	cudaErrorInvalidDevice,
	cudaErrorNotImplemented // Attention: This is not standard CUDA!
}cudaError_t;

enum cudaComputeMode
{
	cudaComputeModeDefault,
	cudaComputeModeExclusive,
	cudaComputeModeProhibited
};

enum cudaMemcpyKind
{
	cudaMemcpyHostToHost,
	cudaMemcpyHostToDevice,
	cudaMemcpyDeviceToHost,
	cudaMemcpyDeviceToDevice,
	cudaMemcpyDefault
};

struct dim3
{
	dim3(float x, float y, float z)
		: x(x), y(y), z(z) {}

	dim3(float x, float y)
		: x(x), y(y), z(0) {}

	dim3(float x)
		: x(x), y(0), z(0) {}

	dim3() : x(0), y(0), z(0) {}

	float x;
	float y;
	float z;
};

typedef struct
{

}cudaEvent_t;

typedef int cudaStream_t;

struct cudaDeviceProp 
{
	char name[256];
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	size_t totalConstMem;
	int major;
	int minor;
	int clockRate;
	size_t textureAlignment;
	int deviceOverlap;
	int multiProcessorCount;
	int kernelExecTimeoutEnabled;
	int l2CacheSize;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int concurrentKernels;
	int ECCEnabled;
	int pciBusID;
	int pciDeviceID;
	int tccDriver;
};

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t err);
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize();

/// Non-CUDA functions
// We don't need to actually copy the arguments since OpenCL does not keep the pointer
// and copies the data for itself.
#define CU_KERNEL_ARG(x) {sizeof(x), cu::addressOf<decltype(x)>(x)}

namespace cu
{

static const int CUDALIBRE_MAJOR = 2;
static const int CUDALIBRE_MINOR = 0;

void initCudaLibre(const char* sources);
void resetCudaLibre();

template<typename T> void* addressOf(const T& src) { return (void*) &src; }
typedef std::vector<std::pair<size_t, void*>> ArgumentList;

/**
 * @brief Calls a kernel in the currently loaded program on the current device.
 *
 * @param name The name of the function to call
 * @param w The "width" of one thread block
 * @param h The "height" of one thread block
 * @param args A list of arguments
 */
bool lcCallKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const ArgumentList& args);
bool lcCallKernel(const char* name, const dim3& gridsize, const dim3& blocksize); // No args
}

#endif
