cmake_minimum_required(VERSION 2.6)
project(CudaLibre)

find_path(LibClang_INCLUDE_DIR clang-c/Index.h)
find_library(LIBCLANG_LIBRARY NAMES clang)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED clangTooling libClang)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

bison_target(parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
flex_target(lexer lexer.l  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)
add_flex_bison_dependency(lexer parser)

add_executable(culcc
		${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS} 
		main.cpp
		ClangTool.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIR})
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
)

#if(${CMAKE_BUILD_TYPE} STREQUAL "Release" OR USE_LOCAL_INCLUDES)
#    add_definitions(-DCUDALIBRE_STANDARD_INCLUDES="${LibreCuda_INCLUDE_DIRS}")
#else()
#    add_definitions(-DCUDALIBRE_STANDARD_INCLUDES="${CMAKE_INSTALL_PREFIX}/include/cudalibre")
#endif()

install(TARGETS culcc
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

add_subdirectory(test)
