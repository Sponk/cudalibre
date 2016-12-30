#pragma once

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
#if !defined(__CUDACC__) || defined(__CUDALIBRE_CLANG__)
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

/// @attention Should not be defined on host code!
extern int get_num_groups(int);
extern int get_local_size(int);
extern int get_group_id(int);
extern int get_local_id(int);

#define __kernel __attribute__((annotate("kernel")))
#define __local __attribute__((annotate("local")))

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;

#endif
