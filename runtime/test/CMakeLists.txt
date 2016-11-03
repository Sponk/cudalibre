find_package(GTest REQUIRED)

include_directories(
		${GTEST_INCLUDE_DIRS}
		${OpenCL_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/runtime
)

add_executable(runtime-tests main.cpp EnumerationTest.cpp KernelTest.cpp)
target_link_libraries(runtime-tests clruntime ${GTEST_LIBRARIES} pthread)
