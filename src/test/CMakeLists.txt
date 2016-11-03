find_package(GTest REQUIRED)

include_directories(
		${GTEST_INCLUDE_DIRS}
)

add_executable(clcc-tests main.cpp ParserTest.cpp  ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS})

add_dependencies(clcc-tests lexer parser)
target_link_libraries(clcc-tests ${GTEST_LIBRARIES} pthread)
