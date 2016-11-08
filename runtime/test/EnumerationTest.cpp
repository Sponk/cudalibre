#include <gtest/gtest.h>
#include <cudalibre.h>
#include <iostream>

using namespace std;

class EnumerationTest: public ::testing::Test
{
protected:
	virtual void TearDown()
	{
		cu::resetCudaLibre();
	}

	virtual void SetUp()
	{
		cu::initCudaLibre("");
	}
};

TEST_F(EnumerationTest, DeviceCount)
{
	// Checks only if the value was changed
	int num = 0xFFFFFF;
	ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&num));
	EXPECT_NE(0xFFFFFF, num); // Can't check for a specific number.
}

TEST_F(EnumerationTest, DeviceCheck) // This test needs at least one existing LibreCUDA (OpenCL) device!
{
	// Checks only if the value was changed
	int num = 0;
	ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&num));
	ASSERT_GE(num, 1);
	
	cudaDeviceProp prop;
	ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&prop, 0));
}
