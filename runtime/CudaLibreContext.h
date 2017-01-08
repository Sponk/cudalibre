#ifndef CUDALIBRE_CUDALIBRECONTEXT_H
#define CUDALIBRE_CUDALIBRECONTEXT_H

#include "CudaDevice.h"
#include <vector>
#include <cassert>
#include <sstream>
#include <queue>
#include <iostream>

namespace cu
{
/**
 * @brief Represents the current system state of CudaLibre.
 *
 * It manages all devices and device initialization.
 */
class CudaLibreContext
{
	CudaDevice* currentDevice;
	std::vector<CudaDevice> cudaDevices;
	std::stringstream sources;
	cl::Context clcontext;

	std::vector<std::pair<const unsigned char*, size_t>> binaries;
	std::vector<std::pair<std::string, int>> sourceList;

public:
	CudaLibreContext();
	~CudaLibreContext()
	{
		clear();
	}

	/**
	 * @brief Sets the current device.
	 * @param id The device number to use.
	 * @return The error code or cudaSuccess.
	 */
	cudaError_t setCurrentDevice(size_t id)
	{
		if(id >= cudaDevices.size())
			return cudaErrorInvalidDevice;
		currentDevice = &cudaDevices[id];
		return cudaSuccess;
	}

	/**
	 * @brief Fetches the number of compute devices.
	 * @return The number of devices.
	 */
	size_t getNumDevices() { return cudaDevices.size(); }

	/**
	 * @brief Fetches the device properties of the given device.
	 * @param prop A device property structure, the destination.
	 * @param device The device to fetch the info from.
	 * @return cudaSuccess or the error code.
	 */
	cudaError_t getDeviceProperties(struct cudaDeviceProp* prop, int device);

	/**
	 * @brief Fetches a reference to the currently selected device.
	 * @return The currently selected device.
	 */
	CudaDevice& getCurrentDevice() { return *currentDevice; }

	/**
	 * @brief Adds more sources to the full kernel code.
	 * @param src The source string.
	 */
	void addSources(const char* src, int priority)
	{
		sources.str("");

		if(sourceList.size() == 0 || sourceList.back().second <= priority)
		{
			for(auto& p : sourceList)
				sources << p.first;

			sourceList.push_back(std::pair<std::string, int>(src, priority));
			sources << src;
		}
		else
			for(int i = 0; i < sourceList.size(); i++)
			{
				if(sourceList[i].second > priority)
				{
					sourceList.insert(sourceList.begin() + i, std::pair<std::string, int>(src, priority));
					sources << src;
				}
				else
				{
					sources << sourceList[i].first;
				}
			}

		for(auto& d : cudaDevices)
			d.setSources(sources.str().c_str());
	}
	
	void addBinary(const unsigned char* src, size_t size)
	{
		binaries.push_back(std::pair<const unsigned char*, size_t>(src, size));
		for(auto& d : cudaDevices)
			d.addBinary(src, size);
	}

	/**
	 * @brief Resets the full kernel code.
	 */
	void clearSources()
	{
		sourceList.clear();
		sources.str("");
		for(auto& d : cudaDevices)
			d.setSources("");
	}

	void clear()
	{
		clearSources();
		for(auto& d : cudaDevices)
			d.clear();
	}

	cl::Context& getContext() { return clcontext; }

	/**
	 * @brief Fetches the currently used kernel code.
	 * @return The kernel source string.
	 */
	std::string getSources() { return sources.str(); }
};

/**
 * @brief Converts OpenCL error codes to CUDA errors.
 * @param err The CL error code.
 * @return The CUDA error code.
 */
cudaError_t clerr2cuderr(int err);

/**
 * @brief Checks if err is a legitimate error and prints a message about it.
 * @param err The error code.
 * @param name The string to print in front of the error code.
 * @return True if err is an error, false if not.
 */
bool checkErr(cl_int err, const char* name);

}

#endif //CUDALIBRE_CUDALIBRECONTEXT_H
