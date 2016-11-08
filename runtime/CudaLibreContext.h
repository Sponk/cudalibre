#ifndef CUDALIBRE_CUDALIBRECONTEXT_H
#define CUDALIBRE_CUDALIBRECONTEXT_H

#include "CudaDevice.h"
#include <vector>
#include <cassert>
#include <sstream>

namespace cu
{
class CudaLibreContext
{
	CudaDevice* currentDevice;
	std::vector<CudaDevice> cudaDevices;
	std::stringstream sources;
	cl::Context clcontext;

public:
	CudaLibreContext();

	cudaError_t setCurrentDevice(size_t id)
	{
		if(id >= cudaDevices.size())
			return cudaErrorInvalidDevice;
		currentDevice = &cudaDevices[id];
		return cudaSuccess;
	}

	size_t getNumDevices() { return cudaDevices.size(); }
	cudaError_t getDeviceProperties(struct cudaDeviceProp* prop, int device);

	CudaDevice& getCurrentDevice() { return *currentDevice; }

	void addSources(const char* src)
	{
		sources << src;
		for(auto& d : cudaDevices)
			d.setSources(sources.str().c_str());
	}

	void clearSources()
	{
		sources.str("");
		for(auto& d : cudaDevices)
			d.setSources("");
	}

	std::string getSources() { return sources.str(); }
};

cudaError_t clerr2cuderr(int err);
bool checkErr(cl_int err, const char* name);

}

#endif //CUDALIBRE_CUDALIBRECONTEXT_H
