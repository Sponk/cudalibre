
cmake_minimum_required(VERSION 3.5)

#set(LC_BINARY_OUTPUT ${CMAKE_BINARY_DIR}/BinaryOutput)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LC_BINARY_OUTPUT})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LC_BINARY_OUTPUT})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LC_BINARY_OUTPUT})

set(CMAKE_MODULE_PATH
        ${CMAKE_SOURCE_DIR}/CMake
        ${CMAKE_SOURCE_DIR}/CMake/Packages
        ${CMAKE_MODULE_PATH}
        )

set(CMAKE_CXX_STANDARD 11)

set(CLCC ${CMAKE_BINARY_DIR}/src/lccc)
include("CudaLibre")

add_subdirectory(src)
add_subdirectory(runtime)
add_subdirectory(test)
add_subdirectory(examples)