#pragma once

#ifndef __CUDA_LIBRE_TRANSLATION_PHASE__

enum cudaChannelFormatKind
{
	cudaChannelFormatKindSigned = 0,
	cudaChannelFormatKindUnsigned = 1,
	cudaChannelFormatKindFloat = 2,
	cudaChannelFormatKindNone = 3
};

struct cudaChannelFormatDesc
{
	int x, y, z, w;
	enum cudaChannelFormatKind f;
};
#endif
