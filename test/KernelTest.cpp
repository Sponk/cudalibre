#include <gtest/gtest.h>
#include <librecuda.h>
#include <iostream>

using namespace std;

TEST(KernelTest, PrintfKernel)
{
	lcSetSources("#pragma OPENCL EXTENSION cl_amd_printf : enable \n__kernel void test() { "
	 "printf(\""
					 "[ RUN      ] KernelTest.KernelPrintf%d\\n"
	                 "[       OK ] KernelTest.KernelPrintf%d (0 ms)\\n\",  get_local_id(0), get_local_id(0));"

	"}\n");

	std::cout << std::endl;
	lcCallKernel("test", 2, 2);
	lcWaitForKernel();
}

TEST(KernelTest, PrintfKernelArg)
{
	lcSetSources("#pragma OPENCL EXTENSION cl_amd_printf : enable \n__kernel void test(int k) { "
					 "printf(\""
					 "[ RUN      ] KernelTest.KernelPrintfArg%d\\n"
					 "[       OK ] KernelTest.KernelPrintfArg%d (0 ms)\\n\",  k, k);"

					 "}\n");

	std::cout << std::endl;

	//lcArgumentList args({{sizeof(5), copyElement<typeof(5)>(5)}});
	lcCallKernel("test", 2, 2, lcArgumentList({LC_KERNEL_ARG(12)}));

	//lcCallKernel("test", 2, 2, args);
	lcWaitForKernel();
}