#pragma once

/// Contains mathematical functions as described by nVidia in
/// http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifdef __CUDA_LIBRE_TRANSLATION_PHASE__

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

extern int get_num_groups(int);
extern int get_local_size(int);
extern int get_group_id(int);
extern int get_local_id(int);

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;

#ifndef __kernel
#define __kernel __attribute__((annotate("kernel")))
#endif

#ifndef __local
#define __local __attribute__((annotate("local")))
#endif
#endif

// Some stuff used in the compiler to prevent errors while transforming code
// @note These won't appear in the OpenCL code since they are defined as macros. THIS IS NECESSARY! DON'T CHANGE!
#define __DEFINE_VECSTRUCT2(type, name) struct name { type x; type y; };
#define __DEFINE_VECSTRUCT3(type, name) struct name { type x; type y; type z;};
#define __DEFINE_VECSTRUCT4(type, name) struct name { type x; type y; type z; type w;};

__DEFINE_VECSTRUCT2(float, float2)
__DEFINE_VECSTRUCT3(float, float3)
__DEFINE_VECSTRUCT4(float, float4)

__DEFINE_VECSTRUCT2(int, int2)
__DEFINE_VECSTRUCT3(int, int3)
__DEFINE_VECSTRUCT4(int, int4)

__DEFINE_VECSTRUCT2(unsigned int, uint2)
__DEFINE_VECSTRUCT3(unsigned int, uint3)
__DEFINE_VECSTRUCT4(unsigned int, uint4)

#ifdef __CUDACC__

#define _CL_BUILTIN_ __attribute__((annotate("builtin")))

_CL_BUILTIN_ __device__ extern float abs(float);
_CL_BUILTIN_ __device__ extern float fabs(float);
_CL_BUILTIN_ __device__ extern float sqrtf(float);
_CL_BUILTIN_ __device__ extern float floorf(float);

_CL_BUILTIN_ __device__ extern float fminf(float a, float b);
_CL_BUILTIN_ __device__ extern float fmaxf(float a, float b);
_CL_BUILTIN_ __device__ extern int max(int a, int b);
_CL_BUILTIN_ __device__ extern int min(int a, int b);
_CL_BUILTIN_ __device__ extern float rsqrtf(float x);
_CL_BUILTIN_ __device__ extern float fmodf(float a, float b);

#undef _CL_BUILTIN_

#else
#include <math.h>
#define __device__
#define __host__
#endif

__host__ __device__ float2 make_float2(float x, float y);
__host__ __device__ float3 make_float3(float x, float y, float z);
__host__ __device__ float4 make_float4(float x, float y, float z, float w);

__host__ __device__ int2 make_int2(int x, int y);
__host__ __device__ int3 make_int3(int x, int y, int z);
__host__ __device__ int4 make_int4(int x, int y, int z, int w);

__host__ __device__ uint2 make_uint2(unsigned int x, unsigned int y);
__host__ __device__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z);
__host__ __device__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);
