find_package(OpenCL REQUIRED)

if(USE_CL1)
  add_definitions(-DUSE_CL1)
endif()

add_library(clruntime SHARED cudalibre.cpp cudalibre.h)
target_link_libraries(clruntime ${OpenCL_LIBRARIES})

set_target_properties(clruntime PROPERTIES PUBLIC_HEADER "cudalibre.h;cuda_runtime.h")

install(TARGETS clruntime
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/cudalibre"
        COMPONENT dev)