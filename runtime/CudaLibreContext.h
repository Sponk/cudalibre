#ifndef CUDALIBRE_CUDALIBRECONTEXT_H
#define CUDALIBRE_CUDALIBRECONTEXT_H

#include "CudaDevice.h"
#include <vector>
#include <cassert>
#include <sstream>

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

public:
	CudaLibreContext();

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
	void addSources(const char* src)
	{
		sources << src;
		for(auto& d : cudaDevices)
			d.setSources(sources.str().c_str());
	}

	/**
	 * @brief Resets the full kernel code.
	 */
	void clearSources()
	{
		sources.str("");
		for(auto& d : cudaDevices)
			d.setSources("");
	}

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
