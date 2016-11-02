#include <gtest/gtest.h>
#include <cudalibre.h>
#include <iostream>

using namespace std;

TEST(KernelTest, PrintfKernel)
{
	ASSERT_TRUE(lcSetSources("//#pragma OPENCL EXTENSION cl_intel_printf : enable \n__kernel void test() { "
	 "printf(\""
					 "[ RUN      ] KernelTest.KernelPrintf%d\\n"
	                 "[       OK ] KernelTest.KernelPrintf%d (0 ms)\\n\",  get_local_id(0), get_local_id(0));"

							 "}\n"));

	std::cout << std::endl;
	EXPECT_TRUE(lcCallKernel("test", 2, 2));
	lcWaitForKernel();
}

TEST(KernelTest, PrintfKernelArg)
{
	ASSERT_TRUE(lcSetSources("//#pragma OPENCL EXTENSION cl_intel_printf : enable \n__kernel void test(int k, float l, float d) { "
					 "printf(\""
					 "[ RUN      ] KernelTest.KernelPrintfArg%d\\n"
					 "[       OK ] KernelTest.KernelPrintfArg%d (0 ms)\\n\",  k, k);"
		             "printf(\"\\nArguments: float %f, float %f\\n\", l, d);"
							 "}\n"));

	std::cout << std::endl;

	EXPECT_TRUE(lcCallKernel("test", 2, 2, lcArgumentList({LC_KERNEL_ARG(12), LC_KERNEL_ARG(42.0f), LC_KERNEL_ARG(3.1415f)})));
	lcWaitForKernel();
}
