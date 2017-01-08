#pragma once

#include <chrono>

/**
 * @defgroup cuda_runtime CUDA Runtime
 * @addtogroup cuda_runtime
 *  @{
 */

#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04

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

#include "cuda_types.h"

typedef struct
{
	std::chrono::high_resolution_clock::time_point* time;
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
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaSetDevice(int device);

/**
 * @brief Synchronizes with the device.
 *
 * It blocks until all device actions are finished.
 *
 * @return cudaSuccess
 */
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaThreadSynchronize();

/// Only needed so the compiler does not fail
extern int cudaConfigureCall(	dim3 gridDim,
				dim3 blockDim,
				int sharedMem = 0,
				int stream = 0);

/// Ensure all math capabilities can be used on the host
/// Note: math.cuh is explicitely written for the purpose of being valid on the host and the device!
#include <math.cuh>

// Ensure some macros are defined to empty
#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __global__
#define __global__
#endif

/**
 * @}
 */
