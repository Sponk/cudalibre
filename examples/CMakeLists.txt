
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

cuda_add_executable(cuinfo cuinfo.cu)