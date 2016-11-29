
include_directories(${CMAKE_SOURCE_DIR}/runtime ${CMAKE_CURRENT_SOURCE_DIR})
cuda_add_executable(VectorPitchTest VectorPitchTest.cu)
add_test(VectorPitchTest VectorPitchTest)

cuda_add_executable(VectorTest VectorTest.cu)
add_test(VectorTest VectorTest)