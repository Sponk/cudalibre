cmake_minimum_required(VERSION 2.6)
project(CudaLibre)

find_path(LibClang_INCLUDE_DIR clang-c/Index.h)
find_library(LIBCLANG_LIBRARY NAMES clang)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED clangTooling libClang)
add_definitions(-DRTINC="-I${CMAKE_SOURCE_DIR}/runtime" -DSTDINC="-I${CMAKE_SOURCE_DIR}/cudastd")

add_executable(culcc
		ClangTool.cpp
		ClangCLProcessor.cpp
		Spir.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIR} ${CudaLibre_INCLUDE_DIRS})
target_link_libraries(culcc
  clangFrontend
  clangSerialization
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangBasic
  clangEdit
  clangLex
  clangTooling
  clruntime
)

install(TARGETS culcc
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

add_subdirectory(test)
