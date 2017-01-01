#pragma once

// Some stuff used in the compiler to prevent errors while transforming code
#if !defined(__CUDACC__) || defined(__CUDALIBRE_OPENCL_EMULATION__)
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

#endif

#ifndef __CUDACC__ // Ensure constructors are available for GCC
#define __device__
#define __host__

#include <cudalibre_runtime.cuh>

#undef __device__
#undef __host__
#endif
