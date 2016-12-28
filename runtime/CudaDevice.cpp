#include "CudaDevice.h"
#include "CudaLibreContext.h"
#include <iostream>
#include <numeric>

using namespace std;
using cu::checkErr;
using cu::clerr2cuderr;

cu::CudaDevice::CudaDevice(cl::Context& context, cl::Device& device, const std::string& kernelcode)
	:
	context(context),
	device(device),
	kernelcode(kernelcode),
	queue(context, device, 0, NULL)
{

}

cu::CudaDevice::CudaDevice(cl::Context& context, cl::Device& device)
	:
	context(context),
	device(device),
	queue(context, device, 0, NULL)
{

}

void cu::CudaDevice::setSources(const char* sources)
{
	kernelCompiled = false;
	kernelcode = sources;
}

cudaError_t cu::CudaDevice::buildKernel(const char* sources)
{
#if !defined(USE_CL1) && !defined(WIN32)
	cl::Program::Sources source(1, sources); //source(1, std::make_pair(sources, strlen(sources)));
#else
	cl::Program::Sources source(1, std::make_pair(sources, strlen(sources)));
#endif

	program = cl::Program(context, source);
	int err = program.build({device}, "-x spir");

	if(checkErr(err, sources))
	{
		cerr << "Build log:" << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
		err = cudaErrorInitializationError;
	}

	return clerr2cuderr(err);
}

void cu::CudaDevice::getProgramBinaries(std::vector<unsigned char>& data)
{
	const vector<size_t> sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
	vector<char*> programs = program.getInfo<CL_PROGRAM_BINARIES>();

	data.resize(sizes[0]);
	std::copy(programs[0], programs[0] + sizes[0], data.begin());
}

static inline void calculateWorksize(const dim3& gridsize, const dim3& blocksize,
									 	cl::NDRange& local, cl::NDRange& global)
{
	//DEBUG("localWork %d %d %d globalWork %d %d %d",
	//	  blocksize.x, blocksize.y, blocksize.z,
	//	  blocksize.x * gridsize.x, blocksize.y * gridsize.y, blocksize.z * gridsize.z);

	if (blocksize.y == 0.0f)
	{
		local = cl::NDRange(blocksize.x);
	}
	else if (blocksize.z == 0.0f)
	{
		local = cl::NDRange(blocksize.x, blocksize.y);
	}
	else
		local = cl::NDRange(blocksize.x, blocksize.y, blocksize.z);

	if (gridsize.y == 0.0f)
	{
		global = cl::NDRange(blocksize.x * gridsize.x);
	}
	else if (gridsize.z == 0.0f)
	{
		global = cl::NDRange(blocksize.x * gridsize.x, blocksize.y * gridsize.y);
	}
	else
		global = cl::NDRange(blocksize.x * gridsize.x, blocksize.y * gridsize.y, blocksize.z * gridsize.z);
}

cudaError_t cu::CudaDevice::callKernel(const char* name, const dim3& gridsize, const dim3& blocksize, const cu::ArgumentList& args)
{
	int err = 0;

	// Make sure the kernelcode has been built
	if(!hasKernel())
		buildKernel(kernelcode.c_str());

	// Check if the kernel is cached or needs to be built
	cl::Kernel* kernel = nullptr;
	auto iterator = kernels.find(name);
	if(iterator != kernels.end())
	{
		kernel = &iterator->second;
	}
	else
	{
		kernel = &kernels[name];
		*kernel = cl::Kernel(program, name, &err);

		if(checkErr(err, "Kernel::Kernel()"))
		{
			// Roll back changes on error
			kernels.erase(name);
			return clerr2cuderr(err);
		}
	}

	unsigned int i = 0;
	for(auto p : args)
	{
		// Check if variable is a valid handle
		auto bufiter = bufferHeap.find(*static_cast<void**>(p.second));
		if(bufiter != bufferHeap.end())
			kernel->setArg(i++, *bufiter->second.buffer);
		else
			kernel->setArg(i++, p.first, p.second);
	}

	cl::NDRange localWork;
	cl::NDRange globalWork;
	cl::Event event;

	calculateWorksize(gridsize, blocksize, localWork, globalWork);

	err = queue.enqueueNDRangeKernel(
		*kernel,
		cl::NullRange, // Has to be NULL
		globalWork,
		localWork,
		NULL,
		&event);

	checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");
	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::mallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
	int err = 0;

	cl::Buffer* buf = new cl::Buffer(context, CL_MEM_READ_WRITE, width * height, nullptr, &err);
	bufferHeap[buf] = UnifiedBuffer(nullptr, buf);

	checkErr(err, "Buffer::Buffer()");

	*pitch = width; // FIXME: Setting pitch to the simple value. Should figure out the HW optimal value!
	*devPtr = (void*) buf; // Hand out the handle to the buffer

	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::malloc(void** devPtr, size_t size)
{
	int err = 0;

	cl::Buffer* buf = new cl::Buffer(context, CL_MEM_READ_WRITE, size, nullptr, &err);
	bufferHeap[buf] = UnifiedBuffer(nullptr, buf);

	checkErr(err, "Buffer::Buffer()");
	*devPtr = (void*) buf; // Hand out the handle to the buffer

	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::mallocManaged(void** devPtr, size_t size, unsigned int flags)
{
	int err = 0;

	cl::Buffer* buf = new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
	void* mem = queue.enqueueMapBuffer(*buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, nullptr, nullptr, &err);
	bufferHeap[mem] = UnifiedBuffer(mem, buf);

	checkErr(err, "Buffer::Buffer()");
	*devPtr = mem; // Hand out the handle to the buffer

	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::memcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
{
	int err = 0;

	auto bufiter = bufferHeap.find(dst);
	if(bufiter == bufferHeap.end())
		return cudaErrorMemoryAllocation;

	switch(kind)
	{
		case cudaMemcpyHostToDevice: // FIXME: Should this be blocking or asynchronous?
			err = queue.enqueueWriteBuffer(*bufiter->second.buffer, CL_TRUE, 0, width * height, src, NULL, NULL);
			break;

		case cudaMemcpyDeviceToHost:
			err = queue.enqueueReadBuffer(*bufiter->second.buffer, CL_TRUE, 0, width * height, (void*) src, NULL, NULL);
			break;

		default: return cudaErrorNotImplemented;
	}

	checkErr(err, "memcpy2D");
	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
	int err = 0;

	auto bufiter = bufferHeap.find(dst);
	if(bufiter == bufferHeap.end())
		return cudaErrorMemoryAllocation;

	switch(kind)
	{
		case cudaMemcpyHostToDevice: // FIXME: Should this be blocking or asynchronous?
			err = queue.enqueueWriteBuffer(*bufiter->second.buffer, CL_TRUE, 0, count, src, NULL, NULL);
			break;

		case cudaMemcpyDeviceToHost:
			err = queue.enqueueReadBuffer(*bufiter->second.buffer, CL_TRUE, 0, count, (void*) src, NULL, NULL);
			break;

		default: return cudaErrorNotImplemented;
	}

	checkErr(err, "memcpy");
	return clerr2cuderr(err);
}

cudaError_t cu::CudaDevice::free(void* ptr)
{
	auto bufiter = bufferHeap.find(ptr);
	if(bufiter == bufferHeap.end())
		return cudaErrorMemoryAllocation;

	// It's a managed buffer!
	if(bufiter->second.host != nullptr)
	{
		queue.enqueueUnmapMemObject(*bufiter->second.buffer, bufiter->second.host);
	}

	delete bufiter->second.buffer;
	bufferHeap.erase(bufiter);
	return cudaSuccess;
}