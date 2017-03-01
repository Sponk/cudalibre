#pragma once

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

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include "vector_types.h"
#include "vector_functions.h"
#include "math_functions.h"

#endif // __CUDA_LIBRE_TRANSLATION_PHASE__
