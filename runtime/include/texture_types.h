#pragma once

#include "driver_types.h"

#define cudaTextureType1D 0x01
#define cudaTextureType2D 0x02
#define cudaTextureType3D 0x03
#define cudaTextureTypeCubemap 0x0C
#define cudaTextureType1DLayered 0xF1
#define cudaTextureType2DLayered 0xF2
#define cudaTextureTypeCubemapLayered 0xFC

#ifndef __CUDA_LIBRE_TRANSLATION_PHASE__

typedef unsigned long long cudaTextureObject_t;

enum cudaTextureFilterMode
{
	cudaFilterModePoint = 0,
	cudaFilterModeLinear = 1
};

enum cudaTextureReadMode
{
	cudaReadModeElementType = 0,
	cudaReadModeNormalizedFloat = 1
};

enum cudaTextureAddressMode
{
	cudaAddressModeWrap = 0,
	cudaAddressModeClamp = 1,
	cudaAddressModeMirror = 2,
	cudaAddressModeBorder = 3
};

struct textureReference
{
	int normalized;
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureAddressMode addressMode[3];
	struct cudaChannelFormatDesc channelDesc;
	int sRGB;
	unsigned int maxAnisotropy;
	enum cudaTextureFilterMode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
	int __cudaReserved[15];
};

struct cudaTextureDesc
{
	enum cudaTextureAddressMode addressMode[3];
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureReadMode readMode;
	int sRGB;
	float borderColor[4];
	int normalizedCoords;
	unsigned int maxAnisotropy;
	enum cudaTextureFilterMode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
};

#endif
