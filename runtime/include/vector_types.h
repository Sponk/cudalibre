#pragma once

//#ifndef __CUDA_LIBRE_TRANSLATION_PHASE__

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

__DEFINE_VECSTRUCT2(unsigned char, uchar2)
__DEFINE_VECSTRUCT3(unsigned char, uchar3)
__DEFINE_VECSTRUCT4(unsigned char, uchar4)

//#endif