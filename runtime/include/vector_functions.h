#pragma once

#include "vector_types.h"

__host__ __device__ float2 make_float2(float x, float y);
__host__ __device__ float3 make_float3(float x, float y, float z);
__host__ __device__ float4 make_float4(float x, float y, float z, float w);

__host__ __device__ int2 make_int2(int x, int y);
__host__ __device__ int3 make_int3(int x, int y, int z);
__host__ __device__ int4 make_int4(int x, int y, int z, int w);

__host__ __device__ uint2 make_uint2(unsigned int x, unsigned int y);
__host__ __device__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z);
__host__ __device__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

__host__ __device__ uchar2 make_uchar2(unsigned char x, unsigned char y);
__host__ __device__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z);
__host__ __device__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);
