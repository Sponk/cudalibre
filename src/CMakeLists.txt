cmake_minimum_required(VERSION 2.6)
project(lccc)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

bison_target(parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
flex_target(lexer lexer.l  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)
add_flex_bison_dependency(lexer parser)

add_executable(clcc ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS} main.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#if(${CMAKE_BUILD_TYPE} STREQUAL "Release" OR USE_LOCAL_INCLUDES)
#    add_definitions(-DCUDALIBRE_STANDARD_INCLUDES="${LibreCuda_INCLUDE_DIRS}")
#else()
#    add_definitions(-DCUDALIBRE_STANDARD_INCLUDES="${CMAKE_INSTALL_PREFIX}/include/cudalibre")
#endif()

install(TARGETS clcc
        EXPORT CudaLibreTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

add_subdirectory(test)
