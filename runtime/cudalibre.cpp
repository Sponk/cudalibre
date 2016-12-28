#include "cudalibre.h"
#include "CudaLibreContext.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdarg>

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

#ifdef WIN32
#define STUB cout << "Stub: " << __FUNCTION__ << " " << __FILE__ << " " << __LINE__ << endl
#else
#define STUB cout << "Stub: " << __func__ << " " << __FILE__ << " " << __LINE__ << endl
#endif

using namespace std;
using namespace cu;

static shared_ptr<cu::CudaLibreContext> g_context = nullptr;

#define ENSURE_INIT { if(g_context == nullptr) { g_context = make_shared<cu::CudaLibreContext>(); }} // g_context->addSources(initialSources); }}
#define RETURN_ERROR(x) s_lastError = x; return s_lastError;
static cudaError_t s_lastError = cudaSuccess;

namespace cu
{

void initCudaLibre(const char* sources)
{
	auto context = getCudaLibreContext();
	context->addSources(sources);
}

void initCudaLibreSPIR(const unsigned char* sources, size_t size)
{
	auto context = getCudaLibreContext();
	context->addBinary(sources, size);
	
}

void resetCudaLibre()
{
	g_context->clear();
	g_context = nullptr;
}

shared_ptr<cu::CudaLibreContext> getCudaLibreContext()
{
	ENSURE_INIT;
	return g_context;
}

cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const cu::ArgumentList& args)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().callKernel(name, gridsize, blocksize, args));
}

cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().callKernel(name, gridsize, blocksize));
}
}

cudaError_t cudaGetDeviceCount(int* count)
{
	ENSURE_INIT;
	*count = g_context->getNumDevices();
	RETURN_ERROR(cudaSuccess);
}

// http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/programming-in-opencl/porting-cuda-applications-to-opencl/
/// @todo Not all fields are filled in yet!
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getDeviceProperties(prop, device));
}

cudaError_t cudaSetDevice(int device)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->setCurrentDevice(device));
}

cudaError_t cudaGetLastError()
{
	return s_lastError;
}

const char* cudaGetErrorString(cudaError_t err)
{
	switch(err)
	{
		case cudaSuccess: return "No Error";
		case cudaErrorMemoryAllocation: return "Memory allocation error";
		case cudaErrorInitializationError: return "Initialization error";
		case cudaErrorInvalidDevice: return "Invalid Device error";
		case cudaErrorNotImplemented: return "Not yet implemented"; // Attention: This is not standard CUDA!
	}
	return "Unknown Error";
}

/// @todo is this the right way to implement it?
cudaError_t cudaDeviceSynchronize()
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().deviceSynchronize());
}

cudaError_t cudaThreadSynchronize()
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().deviceSynchronize());
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().mallocPitch(devPtr, pitch, width, height));
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().mallocManaged(devPtr, size));
}

cudaError_t cudaFree(void* devPtr)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().free(devPtr));
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().memcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().malloc(devPtr, size));
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
	ENSURE_INIT;
	RETURN_ERROR(g_context->getCurrentDevice().memcpy(dst, src, count, kind));
}

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
	event->time = new std::chrono::high_resolution_clock::time_point;
	RETURN_ERROR(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	*event.time = chrono::high_resolution_clock::now();
	RETURN_ERROR(cudaSuccess);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	cudaThreadSynchronize();
	*event.time = chrono::high_resolution_clock::now();
	RETURN_ERROR(cudaSuccess);
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
	auto diff = chrono::duration_cast<chrono::nanoseconds>(*end.time - *start.time);
	*ms = static_cast<float>(diff.count()) / 1000.0f / 1000.0f;
	RETURN_ERROR(cudaSuccess);
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	delete event.time;
	RETURN_ERROR(cudaSuccess);
}
