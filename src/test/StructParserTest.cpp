#include <gtest/gtest.h>
#include <iostream>

using namespace std;

bool parseStruct(const string& src);
std::string parseKernel(const std::string& src);
std::string transformCudaClang(const std::string &code);

TEST(StructParserTest, SimpleKernel)
{
	auto result = transformCudaClang("__attribute__((annotate(\"global\"))) void kern(int i, float* f)\n{\n\tf[i] = 32.0f;\n}\n");
}

TEST(StructParserTest, SimpleDeviceFunction)
{
	auto result = transformCudaClang("__attribute__((annotate(\"device\"))) void kern(int i, float* f)\n{\n\tf[i] = 32.0f;\n}\n");
}

TEST(StructParserTest, SimpleStruct)
{
	auto result = transformCudaClang("struct vec3\n{\nfloat x; float y; float z;\n};\n");
}

TEST(StructParserTest, PointerStructClass)
{
	// @todo Expect death for now. Later it should be supported!
	EXPECT_DEATH(transformCudaClang("struct vec3\n{\nfloat* x; float* y; float* z;\n};\n"), "");
	EXPECT_DEATH(transformCudaClang("class vec3\n{\nfloat* x; float* y; float* z;\n};\n"), "");
}

/*
TEST(StructParserTest, KernelCall)
{
	auto result = parseKernel("kernel<<<32, 32>>>(12, 32, 15);");
	//cout << result;
}

TEST(StructParserTest, KernelCallDim)
{
	auto result = parseKernel("kernel<<<dim3(32, 64, 128), 32>>>(12, 32, 15);");
	//cout << result;
}

TEST(StructParserTest, KernelCallDimCalc)
{
	auto result = parseKernel("kernel<<<dim3(32, 64, 128), 32 * 64>>>(12, 32, 15);");
	//cout << result;
}
*/
