include_directories(
		${GTEST_INCLUDE_DIRS}
		${OpenCL_INCLUDE_DIRS}
		${CudaLibre_INCLUDE_DIRS}
)

add_executable(runtime-tests main.cpp EnumerationTest.cpp KernelTest.cpp
		../CudaLibreContext.cpp ../CudaDevice.cpp ../CudaDevice.h)
target_link_libraries(runtime-tests clruntime ${GTEST_LIBRARIES})

# add_test(RuntimeTest runtime-tests)
