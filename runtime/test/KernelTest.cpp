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
	EXPECT_TRUE(lcCallKernel("test", 4, 4));
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

	EXPECT_TRUE(lcCallKernel("test", 4, 4, lcArgumentList({LC_KERNEL_ARG(12), LC_KERNEL_ARG(42.0f), LC_KERNEL_ARG(3.1415f)})));
	lcWaitForKernel();
}

TEST(KernelTest, UpDownTest)
{
	ASSERT_TRUE(lcSetSources("//#pragma OPENCL EXTENSION cl_intel_printf : enable \n__kernel void test(__global float* src, int i) { "
								 "printf(\""
								 "[ RUN      ] KernelTest.KernelPrintfArg%f\\n"
								 "[       OK ] KernelTest.KernelPrintfArg%f (0 ms)\\n\",  src[get_local_id(0)], src[get_local_id(0)]);"
								 "}\n"));

	std::cout << std::endl;

	size_t testArraySize = 64;
	float* array = new float[testArraySize];
	for(int i = 0; i < testArraySize; i++) array[i] = i;

	float* devArray;
	size_t pitch;

	ASSERT_EQ(cudaSuccess, cudaMallocPitch((void**) &devArray, &pitch, sizeof(float), testArraySize));
	ASSERT_EQ(cudaSuccess, cudaMemcpy2D(devArray, pitch, array, sizeof(float), sizeof(float), testArraySize, cudaMemcpyHostToDevice));

	EXPECT_TRUE(lcCallKernel("test", 1, testArraySize, lcArgumentList({LC_KERNEL_ARG(devArray), LC_KERNEL_ARG(1)})));
	lcWaitForKernel();

	delete[] array;
}