
include_directories(${CMAKE_SOURCE_DIR}/runtime ${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_executable(VectorPitchTest VectorPitchTest.cu)
add_test(VectorPitchTest VectorPitchTest)

cuda_add_executable(VectorPointerStructTest VectorPointerStructTest.cu)
add_test(VectorPointerStructTest VectorPointerStructTest)

cuda_add_executable(VectorTest VectorTest.cu)
add_test(VectorTest VectorTest)

cuda_add_executable(VectorManagedTest VectorManagedTest.cu)
add_test(VectorManagedTest VectorManagedTest)

cuda_add_executable(StructTest StructTest.cu)
add_test(StructTest StructTest)

cuda_add_executable(HostDeviceTest HostDeviceTest.cu)
add_test(HostDeviceTest HostDeviceTest)

cuda_add_executable(SIMDTest SIMDTest.cu)
add_test(SIMDTest SIMDTest)

cuda_add_executable(OperatorOverloadingTest OperatorOverloadingTest.cu)
add_test(OperatorOverloadingTest OperatorOverloadingTest)

cuda_add_executable(FunctionOverloadingTest FunctionOverloadingTest.cu)
add_test(FunctionOverloadingTest FunctionOverloadingTest)

cuda_add_executable(StlInheritanceTest StlInheritanceTest.cu)
add_test(StlInheritanceTest StlInheritanceTest)

cuda_add_executable(SharedMemoryTest SharedMemoryTest.cu)
add_test(SharedMemoryTest SharedMemoryTest)

cuda_add_executable(LibstdTest LibstdTest.cu)
add_test(LibstdTest LibstdTest)
target_link_libraries(LibstdTest cudastd)

cuda_add_executable(cuinfo cuinfo.cu)
