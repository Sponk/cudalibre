
include_directories(${CudaLibre_INCLUDE_DIRS})
cuda_add_library(cudastd math.cu)

set_target_properties(cudastd PROPERTIES PUBLIC_HEADER "math.cuh")

install(TARGETS cudastd
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT slib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/cudalibre"
        COMPONENT dev)
