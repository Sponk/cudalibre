find_package(OpenCL REQUIRED)

if(USE_CL1)
  add_definitions(-DUSE_CL1)
endif()

add_library(clruntime SHARED cudalibre.cpp cudalibre.h CudaDevice.cpp CudaDevice.h CudaLibreContext.cpp CudaLibreContext.h)
target_link_libraries(clruntime ${OpenCL_LIBRARIES})
include_directories(${OpenCL_INCLUDE_DIR})

set_target_properties(clruntime PROPERTIES PUBLIC_HEADER "cudalibre.h;cuda_runtime.h;cuda_types.h;cuda.h")

install(TARGETS clruntime
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT slib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/cudalibre"
        COMPONENT dev)

add_subdirectory(test)
