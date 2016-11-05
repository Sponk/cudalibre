#include "cudalibre.h"

#define CL_HPP_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else

#if !defined(USE_CL1) && !defined(WIN32) && !defined(__CYGWIN__)
#include <CL/cl2.hpp>
#else
#include <CL/cl.hpp>
#endif // USE_CL1

#endif

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdarg>

#ifdef WIN32
#define STUB cout << "Stub: " << __FUNCTION__ << " " << __FILE__ << " " << __LINE__ << endl
#else
#define STUB cout << "Stub: " << __func__ << " " << __FILE__ << " " << __LINE__ << endl
#endif

using namespace std;

class LibreCUDAContext
{
public:
	LibreCUDAContext() :
		currentDevice(0)
	{
		cout << "Initializing OpenCL..." << endl;
		
		/// TODO: Make selection of the platform controllable by using an environment variable or similar
		// Find platforms
		cl::Platform::get(&platforms);
		
		std::string name;
		platforms[0].getInfo(CL_PLATFORM_NAME, &name);
		cout << "Found " << platforms.size() << " OpenCL platforms. Using "
			 << name << " (version " << platforms[0].getInfo<CL_PLATFORM_VERSION>()
			 << ") as default." << endl;
		
		// Initialize context
		cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
		clcontext = cl::Context(CL_DEVICE_TYPE_ALL, properties, nullptr, nullptr, nullptr); // FIXME: Check for error!
		
		// Get devices
		devices = clcontext.getInfo<CL_CONTEXT_DEVICES>();
		cout << "Found " << devices.size() << " OpenCL devices. Using "
			 << devices[0].getInfo<CL_DEVICE_NAME>()
			 << " (version " << devices[0].getInfo<CL_DEVICE_VERSION>() << ") as default." << endl;
		
		queue = cl::CommandQueue(clcontext, devices[0], 0, NULL);
	}
	
	~LibreCUDAContext()
	{
	}
	
	cl::CommandQueue queue;
	cl::Program program;
	cl::Context clcontext;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	size_t bufferHeapIndex = 0;
	std::unordered_map<size_t, cl::Buffer> bufferHeap;
	
	size_t currentDevice;
	
} g_context;

inline const char* lcGetErrorString(cl_int error)
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

// TODO: Implement error conversion!
cudaError_t clerr2cuderr(int err)
{
	switch(err)
	{
		case 0: return cudaSuccess;
	}

	return libreCudaErrorNotImplemented;
}

inline bool checkErr(cl_int err, const char* name) 
{
	if(err != CL_SUCCESS) 
	{
		std::cerr << "OpenCL ERROR: " << name << ": " << lcGetErrorString(err)  << " (" << err << ")" << std::endl;
		return true;
	}
	return false;
}

bool lcSetSources(const char* sources)
{
#if !defined(USE_CL1) && !defined(WIN32)
	cl::Program::Sources source(1, sources); //source(1, std::make_pair(sources, strlen(sources) + 1));
#else
	cl::Program::Sources source(1, std::make_pair(sources, strlen(sources) + 1));
#endif
	
	g_context.program = cl::Program(g_context.clcontext, source);
	int err = g_context.program.build(g_context.devices, "");

	if(checkErr(err, sources))
	{
		cout << "Build log:" << endl << g_context.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(g_context.devices[g_context.currentDevice]) << endl;
		return false;
	}

	return true;
}

void lcWaitForKernel()
{
	g_context.queue.finish();
}

static int callKernel(cl::Kernel& kernel, const dim3& gridsize, const dim3& blocksize)
{
	cl::NDRange localWork;
	cl::NDRange globalWork;

	if(blocksize.y == 0.0f)
	{
		localWork = cl::NDRange(blocksize.x);
	}
	else if(blocksize.z == 0.0f)
	{
		localWork = cl::NDRange(blocksize.x, blocksize.y);
	}
	else
		localWork = cl::NDRange(blocksize.x, blocksize.y, blocksize.z);

	if(gridsize.y == 0.0f)
	{
		globalWork = cl::NDRange(blocksize.x * gridsize.x);
	}
	else if(gridsize.z == 0.0f)
	{
		globalWork = cl::NDRange(blocksize.x * gridsize.x, blocksize.y * gridsize.y);
	}
	else
		globalWork = cl::NDRange(blocksize.x * gridsize.x, blocksize.y * gridsize.y, blocksize.z * gridsize.z);

	cl::Event event;
	return g_context.queue.enqueueNDRangeKernel(
		kernel,
		cl::NullRange, // Has to be NULL
		globalWork,
		localWork,
		NULL,
		&event);
}

bool lcCallKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const lcArgumentList& args)
{
	int err;
	cl::Kernel kernel(g_context.program, name, &err);
	checkErr(err, "Kernel::Kernel()");

	int i = 0;
	for(auto p : args)
	{
		// Check if variable is a valid handle
		auto bufiter = g_context.bufferHeap.find(*static_cast<size_t*>(p.second));
		if(bufiter != g_context.bufferHeap.end())
			kernel.setArg(i++, bufiter->second);
		else
			kernel.setArg(i++, p.first, p.second);
	}

	cl::Event event;
	err = callKernel(kernel, gridsize, blocksize);

	return !checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");
}

bool lcCallKernel(const char* name, const dim3& gridsize, const dim3& blocksize)
{
	int err;
	cl::Kernel kernel(g_context.program, name, &err);
	checkErr(err, "Kernel::Kernel()");

	err = callKernel(kernel, gridsize, blocksize);
	return !checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");
}

cudaError_t cudaGetDeviceCount(int* count)
{
	*count = g_context.devices.size();
	return cudaSuccess;
}

// http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/programming-in-opencl/porting-cuda-applications-to-opencl/
/// TODO: Not all fields are filled in yet!
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
	if(device < 0 || device >= g_context.devices.size())
		return cudaErrorInvalidDevice;
	
	// Fill every field with 0 so no uninitialized data is left in there.
	memset(prop, 0, sizeof(struct cudaDeviceProp));
	
	cl::Device& cldev = g_context.devices[device];
	strncpy(prop->name, cldev.getInfo<CL_DEVICE_NAME>().c_str(), sizeof(prop->name));
	
	prop->totalGlobalMem = cldev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	prop->sharedMemPerBlock = cldev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(); // CUDA shared memory <=> OpenCL local memory
	prop->clockRate = cldev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() * 1000; // OpenCL is in MHz, CUDA is in KHz
	prop->totalConstMem = cldev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
	prop->multiProcessorCount = cldev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	prop->computeMode = cudaComputeModeDefault; /// TODO: Always default access?
	
	prop->major = LIBRECUDA_MAJOR;
	prop->minor = LIBRECUDA_MINOR;
	
	prop->concurrentKernels = 0; /// TODO: Is is possible to run kernels concurrently with OpenCL?
	prop->ECCEnabled = 0;
	prop->canMapHostMemory = 0;
	return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
	if(device < 0 || device >= g_context.devices.size())
		return cudaErrorInvalidDevice;
	
	g_context.currentDevice = device;
	return cudaSuccess;
}

cudaError_t cudaGetLastError()
{
	STUB;
	return libreCudaErrorNotImplemented;
}

const char* cudaGetErrorString(cudaError_t err)
{
	STUB;
	return "Unknown Error";
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
	int err = 0;

	cl::Buffer& buf = g_context.bufferHeap[++g_context.bufferHeapIndex];
	buf = cl::Buffer(g_context.clcontext, CL_MEM_READ_WRITE, width * height, NULL, &err);

	*pitch = width; // FIXME: Setting pitch to the simple value. Should figure out the HW optimal value!
	*devPtr = (void*) g_context.bufferHeapIndex; // Hand out the handle to the buffer

	return clerr2cuderr(err);
}

cudaError_t cudaFree(void* devPtr)
{
	auto bufiter = g_context.bufferHeap.find(reinterpret_cast<size_t>(devPtr));
	if(bufiter == g_context.bufferHeap.end())
		return cudaErrorMemoryAllocation;

	g_context.bufferHeap.erase(bufiter);
	return cudaSuccess;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
{
	int err = 0;

	auto bufiter = g_context.bufferHeap.find(reinterpret_cast<size_t>(dst));
	if(bufiter == g_context.bufferHeap.end())
		return cudaErrorMemoryAllocation;

	switch(kind)
	{
		case cudaMemcpyHostToDevice: // FIXME: Should this be blocking or asynchronous?
			err = g_context.queue.enqueueWriteBuffer(bufiter->second, CL_TRUE, 0, width * height, src, NULL, NULL);
			break;

		case cudaMemcpyDeviceToHost:
			err = g_context.queue.enqueueReadBuffer(bufiter->second, CL_TRUE, 0, width * height, dst, NULL, NULL);
			break;

		default: return libreCudaErrorNotImplemented;
	}

	return clerr2cuderr(err);
}

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
	STUB;
	return libreCudaErrorNotImplemented;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	STUB;
	return libreCudaErrorNotImplemented;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	STUB;
	return libreCudaErrorNotImplemented;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
	STUB;
	return libreCudaErrorNotImplemented;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	STUB;
	return libreCudaErrorNotImplemented;
}
