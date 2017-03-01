#pragma once

/// Needs to be in here and in math.cuh since cudastd depends on this lib
/// so depending back is not possible. @fixme Maybe move this code to third, header only lib?
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

#ifdef __CUDALIBRE_CLANG__

/// @attention Should not be defined on host code!
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

#ifndef __host__
#define __host__
#endif

#endif
