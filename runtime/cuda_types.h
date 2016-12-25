#ifndef __CUDA_TYPES_H__
#define __CUDA_TYPES_H__

struct dim3
{
	dim3(int x, int y, int z)
		: x(x), y(y), z(z) {}

	dim3(int x, int y)
		: x(x), y(y), z(0) {}

	dim3(int x)
		: x(x), y(0), z(0) {}

	dim3() : x(0), y(0), z(0) {}

	int x;
	int y;
	int z;
};

// Some stuff used in the compiler to prevent errors while transforming code
#ifdef __CUDALIBRE_CLANG__

/// The device annotation is used to transport definitions to the
/// OpenCL translator but will be removed in the final result
__attribute__((annotate("device"))) dim3 threadIdx;
#endif

#endif