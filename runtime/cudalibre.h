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

#include "cuda_runtime.h"

#define __DEBUG__
#ifdef __DEBUG__
#define DEBUG(format, ...) fprintf(stdout, "Debug in %s at %d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__ )
#else
#define DEBUG(format, ...)
#endif

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
cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const ArgumentList& args);
cudaError_t callKernel(const char* name, const dim3& gridsize, const dim3& blocksize); // No args
}

#endif
