find_package(OpenCL REQUIRED)

if(USE_CL1)
  add_definitions(-DUSE_CL1)
endif()

add_library(clruntime SHARED
        cudalibre.cpp
        cudalibre.h
        CudaDevice.cpp
        CudaDevice.h
        CudaLibreContext.cpp
        CudaLibreContext.h)

target_link_libraries(clruntime ${OpenCL_LIBRARIES})
include_directories(${CudaLibre_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(clruntime PROPERTIES PUBLIC_HEADER
        "cudalibre.h;cuda_runtime.h;cuda_types.h;cuda.h")

install(TARGETS clruntime
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT slib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
        COMPONENT dev)

set(cuda_link   ${CMAKE_INSTALL_PREFIX}/lib64/clruntime.so)
set(cuda_target ${CMAKE_INSTALL_PREFIX}/lib64/cudart.so)
add_custom_command( OUTPUT ${cuda_link}
                    COMMAND ln -s ${cuda_target} ${cuda_link}
                    DEPENDS install ${cuda_target}
                    COMMENT "Generating CUDA symbolic link")

add_custom_target(install_legacy DEPENDS ${cuda_link})

add_subdirectory(test)
