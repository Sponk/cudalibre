
include_directories(${CMAKE_SOURCE_DIR}/runtime)
cuda_add_executable(CudaTest test.cu)

cuda_add_executable(rodinia-gauss rodinia-gauss.cu)
