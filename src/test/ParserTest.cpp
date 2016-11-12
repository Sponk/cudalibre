#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

using namespace std;
extern stringstream cppstream, clstream;

// In parser.y
extern int parse(const char*);

TEST(Parser, GlobalFunction)
{
	clstream.str("");
	EXPECT_FALSE(parse("__global__ void func(int a, int b) {}\n"));
	EXPECT_TRUE(cppstream.str().empty());
	EXPECT_NE(-1, clstream.str().find("__kernel"));
}

TEST(Parser, GlobalVariable)
{
	clstream.str("");
	EXPECT_FALSE(parse("__global__ unsigned int var = 32 + 45;\n"));

	EXPECT_TRUE(cppstream.str().empty());
	EXPECT_EQ(-1, clstream.str().find("__kernel"));
}

TEST(Parser, Device)
{
	clstream.str("");
	EXPECT_FALSE(parse("__device__ void func(int a, int b) {}\n"));
	EXPECT_TRUE(cppstream.str().empty());
	EXPECT_EQ(-1, clstream.str().find("__kernel"));
}

TEST(Parser, Struct)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_FALSE(parse("struct test {\n int l;\nint k;\n};\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_FALSE(clstream.str().empty());

	EXPECT_NE(-1, clstream.str().find("};"));
	EXPECT_NE(-1, cppstream.str().find("};"));
}

TEST(Parser, TypedefStruct)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_FALSE(parse("typedef struct test {\n int l;\nint k\n} test;\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_FALSE(clstream.str().empty());
}

TEST(Parser, Typedef)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_FALSE(parse("typedef int test;\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_FALSE(clstream.str().empty());
}

TEST(Parser, LineComment)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_EQ(0, parse("// typedef int test;\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_TRUE(clstream.str().empty());
}

TEST(Parser, MultilineComment)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_EQ(0, parse("/*\n typedef int test;\n*\\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_TRUE(clstream.str().empty());
}

TEST(Parser, KernelCall)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_EQ(0, parse("kernel<<<32, 12>>>(a, b, c);\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_TRUE(clstream.str().empty());

	EXPECT_STREQ("cu::callKernel(\"kernel\", 32, 12, cu::ArgumentList({CU_KERNEL_ARG(a), CU_KERNEL_ARG( b), CU_KERNEL_ARG( c)}));\n\n"
	, cppstream.str().c_str());

	EXPECT_EQ(0, parse("kernel\t<<<dim3(32), sizeof(12)*5>>>\t(a, b, c);\n"));
}

TEST(Parser, Include)
{
	cppstream.str("");
	clstream.str("");
	EXPECT_FALSE(parse("#include \"test.h\"\n"));
	EXPECT_FALSE(cppstream.str().empty());
	EXPECT_TRUE(clstream.str().empty());
}
