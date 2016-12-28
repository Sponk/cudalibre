#include <cudalibre.h>
#include <CudaLibreContext.h>
#include <CudaDevice.h>
#include <iostream>

using namespace std;

int compileSpir(const std::string& src, std::vector<unsigned char>& program)
{
	auto context = cu::getCudaLibreContext();
	auto device = context->getCurrentDevice();

	device.buildSources(src.c_str());
	device.getProgramBinaries(program);
}
