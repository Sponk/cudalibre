#pragma once

inline __host__ __device__ float2 make_float2(float x, float y) { float2 r; r.x = x; r.y = y; return r; }
inline __host__ __device__ float3 make_float3(float x, float y, float z) { float3 r; r.x = x; r.y = y; r.z = z; return r; }
inline __host__ __device__ float4 make_float4(float x, float y, float z, float w) { float4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

inline __host__ __device__ int2 make_int2(int x, int y) { int2 r; r.x = x; r.y = y; return r; }
inline __host__ __device__ int3 make_int3(int x, int y, int z) { int3 r; r.x = x; r.y = y; r.z = z; return r; }
inline __host__ __device__ int4 make_int4(int x, int y, int z, int w) { int4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

inline __host__ __device__ uint2 make_uint2(unsigned int x, unsigned int y) { uint2 r; r.x = x; r.y = y; return r; }
inline __host__ __device__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 r; r.x = x; r.y = y; r.z = z; return r; }
inline __host__ __device__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { uint4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

