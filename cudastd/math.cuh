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

extern void __syncthreads();
extern void __threadfence_block();

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;

#ifndef __kernel
#define __kernel __attribute__((annotate("kernel")))
#endif

#ifndef __local
#define __local __attribute__((annotate("local")))
#endif

#ifndef __shared__
#define __shared__ __local
#endif

#endif // __CUDA_LIBRE_TRANSLATION_PHASE__

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

/// Used, when its name has to be exactly the same
#define _CL_BUILTIN_ __attribute__((annotate("builtin")))

/// Used, when the name has a CUDA typical trailing 'f' that needs to be cut off.
#define _CL_BUILTINF_ __attribute__((annotate("builtinf")))

/**
 * @defgroup cudastd CUDA Device Standard Library
 * @addtogroup cudastd
 *  @{
 */

_CL_BUILTIN_ __device__ extern int printf(const char* fmt, int);
 
/// @todo Implementation
_CL_BUILTIN_ __device__ extern float abs(float);
_CL_BUILTIN_ __device__ extern float fabs(float);
_CL_BUILTINF_ __device__ extern float sqrtf(float);
_CL_BUILTINF_ __device__ extern float floorf(float);

_CL_BUILTINF_ __device__ extern float fminf(float a, float b);
_CL_BUILTINF_ __device__ extern float fmaxf(float a, float b);
_CL_BUILTIN_ __device__ extern int max(int a, int b);
_CL_BUILTIN_ __device__ extern int min(int a, int b);
_CL_BUILTINF_ __device__ extern float rsqrtf(float x);
_CL_BUILTINF_ __device__ extern float fmodf(float a, float b);
           
_CL_BUILTINF_ __device__ extern float acosf(float x);
_CL_BUILTINF_ __device__ extern float acoshf(float x);
_CL_BUILTINF_ __device__ extern float asinf(float x);
_CL_BUILTINF_ __device__ extern float asinhf(float x);
_CL_BUILTINF_ __device__ extern float atanf(float x);
_CL_BUILTINF_ __device__ extern float cbrtf(float x);
_CL_BUILTINF_ __device__ extern float ceilf(float x);
_CL_BUILTINF_ __device__ extern float copysignf(float x);
_CL_BUILTINF_ __device__ extern float cosf(float x);
_CL_BUILTINF_ __device__ extern float coshf(float x);

/**
 * @brief \f$cospif(x) := cos(\pi * x)\f$
 * @param x Input.
 * @return \f$cos(\pi * x)\f$
 */
_CL_BUILTINF_ __device__ extern float cospif(float x);

_CL_BUILTINF_ __device__ extern float cyl_besseli0f(float x);
_CL_BUILTINF_ __device__ extern float cyl_bessel_i1f(float x);
_CL_BUILTINF_ __device__ extern float erfcinvf(float x);
_CL_BUILTINF_ __device__ extern float erfcf(float x);
_CL_BUILTINF_ __device__ extern float erfcxf(float x);
_CL_BUILTINF_ __device__ extern float erff(float x);
_CL_BUILTINF_ __device__ extern float erfinvf(float x);
_CL_BUILTINF_ __device__ extern float exp10f(float x);
_CL_BUILTINF_ __device__ extern float exp2f(float x);
_CL_BUILTINF_ __device__ extern float expm1f(float x);
_CL_BUILTINF_ __device__ extern float fabsf(float x);
_CL_BUILTINF_ __device__ extern float floorf(float x);
_CL_BUILTINF_ __device__ extern float ilogbf(float x);
_CL_BUILTINF_ __device__ extern float isfinite(float x);
_CL_BUILTINF_ __device__ extern float isinf(float x);
_CL_BUILTINF_ __device__ extern float isnan(float x);
_CL_BUILTINF_ __device__ extern float j0f(float x);
_CL_BUILTINF_ __device__ extern float j1f(float x);
_CL_BUILTINF_ __device__ extern float lgammaf(float x);
_CL_BUILTINF_ __device__ extern long long int llrintf(float x);
_CL_BUILTINF_ __device__ extern long long int llroundf(float x);
_CL_BUILTINF_ __device__ extern float log10f(float x);
_CL_BUILTINF_ __device__ extern float log1pf(float x);
_CL_BUILTINF_ __device__ extern float log2f(float x);
_CL_BUILTINF_ __device__ extern float logbf(float x);
_CL_BUILTINF_ __device__ extern float logf(float x);
_CL_BUILTINF_ __device__ extern float nearbyintf(float x);
_CL_BUILTINF_ __device__ extern float normcdff(float x);
_CL_BUILTINF_ __device__ extern float normcdfinvf(float x);
_CL_BUILTINF_ __device__ extern float normf(float x);
_CL_BUILTINF_ __device__ extern float powf(float x);
_CL_BUILTINF_ __device__ extern float rcbrtf(float x);
_CL_BUILTINF_ __device__ extern float rintf(float x);
_CL_BUILTINF_ __device__ extern float roundf(float x);
_CL_BUILTINF_ __device__ extern float rsqrtf(float x);
_CL_BUILTINF_ __device__ extern float sinf(float x);
_CL_BUILTINF_ __device__ extern float sinhf(float x);
_CL_BUILTINF_ __device__ extern float sinpif(float x);
_CL_BUILTINF_ __device__ extern float sqrtf(float x);
_CL_BUILTINF_ __device__ extern float tanf(float x);
_CL_BUILTINF_ __device__ extern float tanhf(float x);
_CL_BUILTINF_ __device__ extern float tgammaf(float x);
_CL_BUILTINF_ __device__ extern float truncf(float x);
_CL_BUILTINF_ __device__ extern float y0f(float x);
_CL_BUILTINF_ __device__ extern float y1f(float x);

_CL_BUILTINF_ __device__ extern float ynf(int n, float x);
_CL_BUILTINF_ __device__ extern float jnf(int n, float x);

/// Atomics! YEAH!

/**
 * OpenCL Builtins
 */

#ifndef __local
#define __local
#endif

_CL_BUILTIN_ __device__ extern int atomic_add(int* p, int val);
_CL_BUILTIN_ __device__ extern unsigned int atomic_add(unsigned int* p, unsigned int val);

_CL_BUILTIN_ __device__ extern int atomic_inc(int* p);
_CL_BUILTIN_ __device__ extern unsigned int atomic_inc(unsigned int* p);

/**
 * @brief Atomic increment
 *
 * Increases *addr by one, if (val <= *addr).
 *
 * @attention Does not work like in CUDA, val is ignored!
 * @todo Fix this!
 * @return The original value found in *addr.
 */
__device__ int atomicInc(__local int* addr, int val);
__device__ int atomicAdd(__local int* addr, int val);

#undef _CL_BUILTIN_
#undef __DEFINE_VECSTRUCT2
#undef __DEFINE_VECSTRUCT3
#undef __DEFINE_VECSTRUCT4

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

/**
 * @}
 */
