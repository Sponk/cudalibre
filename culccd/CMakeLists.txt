
add_executable(culccd main.cpp)

install(TARGETS culccd
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
