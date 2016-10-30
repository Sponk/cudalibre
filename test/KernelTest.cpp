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
	lcSetSources("#pragma OPENCL EXTENSION cl_amd_printf : enable \n__kernel void test(int k, float l, double d) { "
					 "printf(\""
					 "[ RUN      ] KernelTest.KernelPrintfArg%d\\n"
					 "[       OK ] KernelTest.KernelPrintfArg%d (0 ms)\\n\",  k, k);"
		             "printf(\"\\nArguments: float %f, double %f\", l, d);"
					 "}\n");

	std::cout << std::endl;

	lcCallKernel("test", 2, 2, lcArgumentList({LC_KERNEL_ARG(12), LC_KERNEL_ARG(42.0f), LC_KERNEL_ARG(3.1415)}));
	lcWaitForKernel();
}