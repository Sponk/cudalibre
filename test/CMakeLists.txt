find_package(OpenCL REQUIRED)
find_package(GTest REQUIRED)

include_directories(
		${GTEST_INCLUDE_DIRS}
		${OpenCL_INCLUDE_DIRS}
		../runtime
)

add_executable(tests main.cpp EnumerationTest.cpp KernelTest.cpp)
target_link_libraries(tests lcruntime ${OpenCL_LIBRARIES} ${GTEST_LIBRARIES} pthread)
