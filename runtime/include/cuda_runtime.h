#pragma once

#include <chrono>
#include <cstdlib>

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
	cudaSuccess = 0,
	cudaErrorMissingConfiguration = 1,
	cudaErrorMemoryAllocation = 2,
	cudaErrorInitializationError = 3,
	cudaErrorLaunchFailure = 4,
	cudaErrorPriorLaunchFailure = 5,
	cudaErrorLaunchTimeout = 6,
	cudaErrorLaunchOutOfResources = 7,
	cudaErrorInvalidDeviceFunction = 8,
	cudaErrorInvalidConfiguration = 9,
	cudaErrorInvalidDevice = 10,
	cudaErrorInvalidValue = 11,
	cudaErrorInvalidPitchValue = 12,
	cudaErrorInvalidSymbol = 13,
	cudaErrorMapBufferObjectFailed = 14,
	cudaErrorUnmapBufferObjectFailed = 15,
	cudaErrorInvalidHostPointer = 16,
	cudaErrorInvalidDevicePointer = 17,
	cudaErrorInvalidTexture = 18,
	cudaErrorInvalidTextureBinding = 19,
	cudaErrorInvalidMemcpyDirection = 20,
	cudaErrorAddressOfConstant = 21,
	cudaErrorTextureFetchFailed = 22,
	cudaErrorTextureNotBound = 23,
	cudaErrorSynchronizationError = 24,
	cudaErrorInvalidFilterSetting = 25,
	cudaErrorInvalidNormSetting = 26,
	cudaErrorMixedDeviceExecution = 27,
	cudaErrorCudartUnloading = 28,
	cudaErrorUnknown = 29,
	cudaErrorNotYetImplemented = 30,
	cudaErrorMemoryValueTooLarge = 31,
	cudaErrorInvalidResourceHandle = 32,
	cudaErrorNotReady = 33,
	cudaErrorInsufficientDriver = 34,
	cudaErrorSetOnActiveProcess = 35,
	cudaErrorInvalidSurface = 36,
	cudaErrorNoDevice = 37,
	cudaErrorECCUncorrectable = 38,
	cudaErrorSharedObjectSymbolNotFound = 39,
	cudaErrorSharedObjectInitFailed = 40,
	cudaErrorDuplicateVariableName = 41,
	cudaErrorDuplicateTextureName = 42,
	cudaErrorDuplicateSurfaceName = 43,
	cudaErrorDevicesUnavailable = 44,
	cudaErrorInvalidKernelImage = 45,
	cudaErrorIncompatibleDriverContext = 46,
	cudaErrorPeerAccessAlreadyEnabled = 47,
	cudaErrorPeerAccessNotEnabled = 48,
	cudaErrorDeviceAlreadyInUse = 49,
	cudaErrorProfilerDisabled = 50,
	cudaErrorProfilerNotInitialized = 51,
	cudaErrorProfilerAlreadyStarted = 52,
	cudaErrorProfilerAlreadyStopped = 53,
	cudaErrorAssert = 54,
	cudaErrorTooManyPeers = 55,
	cudaErrorHostMemoryAlreadyRegistered = 56,
	cudaErrorHostMemoryNotRegistered = 57,
	cudaErrorOperatingSystem = 58,
	cudaErrorPeerAccessUnsupported = 59,
	cudaErrorLaunchMaxDepthExceeded = 60,
	cudaErrorLaunchFileScopedTex = 61,
	cudaErrorLaunchFileScopedSurf = 62,
	cudaErrorSyncDepthExceeded = 63,
	cudaErrorLaunchPendingCountExceeded = 64,
	cudaErrorNotPermitted = 65,
	cudaErrorNotSupported = 66,
	cudaErrorHardwareStackError = 67,
	cudaErrorIllegalInstruction = 68,
	cudaErrorMisalignedAddress = 69,
	cudaErrorInvalidAddressSpace = 70,
	cudaErrorInvalidPc = 71,
	cudaErrorIllegalAddress = 72,
	cudaErrorInvalidPtx = 73,
	cudaErrorInvalidGraphicsContext = 74,
	cudaErrorNvlinkUncorrectable = 75,
	cudaErrorStartupFailure = 76,
	cudaErrorApiFailureBase = 77,
	cudaErrorInvalidChannelDescriptor,
	cudaErrorUnsupportedLimit,
	cudaErrorNoKernelImageForDevice,
	cudaErrorNotImplemented = cudaErrorNotYetImplemented// Attention: This is not standard CUDA!
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
cudaError_t cudaDeviceReset();
cudaError_t cudaGetDevice(int* device);

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

#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__
#endif

#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif

#include "vector_functions.h"
#include "math.cuh"

/**
 * @}
 */
