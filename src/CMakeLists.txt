cmake_minimum_required(VERSION 2.6)
project(lccc)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

BISON_TARGET(parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
FLEX_TARGET(lexer lexer.l  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)
ADD_FLEX_BISON_DEPENDENCY(lexer parser)

add_executable(lccc ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS lccc RUNTIME DESTINATION bin)
