#pragma once

#include "texture_types.h"

#ifndef __CUDA_LIBRE_TRANSLATION_PHASE__

template<class T,
	int texType, // = cudaTextureType1D,
	enum cudaTextureReadMode mode> // = cudaReadModeElementType>
struct texture:	public textureReference
{
#ifdef __CUDACC__
	__host__ texture(int normalized = 0,
					 enum cudaTextureFilterMode filterMode = cudaFilterModePoint,
					 enum cudaTextureAddressMode addressMode = cudaAddressModeClamp)
	{
		this->normalized = normalized;
		this->filterMode = filterMode;
		this->channelDesc ;//= cudaCreateChannelDesc<T>();
		this->sRGB = 0;

		this->addressMode[0] = addressMode;
		this->addressMode[1] = addressMode;
		this->addressMode[2] = addressMode;
	}

	__host__ texture(int normalized,
					 enum cudaTextureFilterMode filterMode,
					 enum cudaTextureAddressMode addressMode,
					 struct cudaChannelFormatDesc desc)
	{
		this->normalized = normalized;
		this->filterMode = filterMode;
		this->channelDesc = desc;
		this->sRGB = 0;

		this->addressMode[0] = addressMode;
		this->addressMode[1] = addressMode;
		this->addressMode[2] = addressMode;
	}
#endif // __CUDACC__
};

#endif
