#include "CudaLibreContext.h"
#include <iostream>

using std::cout;
using std::endl;

inline static const char* getErrorString(cl_int error)
{
	switch(error)
	{
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}

namespace cu
{
/// @todo Implement error conversion!
cudaError_t clerr2cuderr(int err)
{
	switch (err)
	{
		case 0: return cudaSuccess;
		case CL_INVALID_KERNEL_NAME: return cudaErrorInitializationError;
	}

	return cudaErrorNotImplemented;
}

bool checkErr(cl_int err, const char* name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "OpenCL ERROR: " << name << ": " << getErrorString(err) << " (" << err << ")" << std::endl;
		return true;
	}
	return false;
}
}

cu::CudaLibreContext::CudaLibreContext()
{
	const char* envPlatform = getenv("CUDALIBRE_PLATFORM");
	const char* envDevice = getenv("CUDALIBRE_DEVICE");

	int device = 0, platform = 0;
	if(envPlatform)
		platform = std::stoi(envPlatform);

	if(envDevice)
		device = std::stoi(envDevice);

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	int err = 0;

	// cout << "Initializing OpenCL..." << endl;

	// Find platforms
	cl::Platform::get(&platforms);

	// FIXME: Shoud I use assertions?
	assert(platform >= 0 && platform < platforms.size());

	std::string name;
	platforms[platform].getInfo(CL_PLATFORM_NAME, &name);
	/*cout << "Found " << platforms.size() << " OpenCL platforms. Using "
		 << name << " (version " << platforms[platform].getInfo<CL_PLATFORM_VERSION>()
		 << ") as default." << endl;*/

	// Initialize context
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platform])(), 0 };
	clcontext = cl::Context(CL_DEVICE_TYPE_ALL, properties, nullptr, nullptr, &err); // FIXME: Check for error!

	if(checkErr(err, "Context::Context()"))
		return;

	// Get devices
	devices = clcontext.getInfo<CL_CONTEXT_DEVICES>();
	assert(device >= 0 && device < devices.size());

	/*cout << "Found " << devices.size() << " OpenCL devices. Using "
		 << devices[device].getInfo<CL_DEVICE_NAME>()
		 << " (version " << devices[device].getInfo<CL_DEVICE_VERSION>() << ") as default." << endl;*/

	for(auto d : devices)
	{
		cudaDevices.push_back(CudaDevice(clcontext, d));
	}

	currentDevice = &cudaDevices[device];
}

cudaError_t cu::CudaLibreContext::getDeviceProperties(struct cudaDeviceProp* prop, int device)
{
	if(device < 0 || device >= getNumDevices())
		return cudaErrorInvalidDevice;

	// Fill every field with 0 so no uninitialized data is left in there.
	memset(prop, 0, sizeof(struct cudaDeviceProp));

	const cl::Device& cldev = cudaDevices[device].getDevice();
	strncpy(prop->name, cldev.getInfo<CL_DEVICE_NAME>().c_str(), sizeof(prop->name));

	prop->l2CacheSize = cldev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
	prop->totalGlobalMem = cldev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	prop->sharedMemPerBlock = cldev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(); // CUDA shared memory <=> OpenCL local memory
	prop->clockRate = cldev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() * 1000; // OpenCL is in MHz, CUDA is in KHz
	prop->totalConstMem = cldev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
	prop->multiProcessorCount = cldev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	prop->computeMode = cudaComputeModeDefault; /// @todo Always default access?

	prop->major = cu::CUDALIBRE_MAJOR;
	prop->minor = cu::CUDALIBRE_MINOR;

	prop->concurrentKernels = 0; /// @todo Is is possible to run kernels concurrently with OpenCL?
	prop->ECCEnabled = 0;
	prop->canMapHostMemory = 0;
	return cudaSuccess;
}