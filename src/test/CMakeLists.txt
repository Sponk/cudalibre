include_directories(
		${GTEST_INCLUDE_DIRS}
)

bison_target(parsertest ../parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
flex_target(lexertest ../lexer.l  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)
add_flex_bison_dependency(lexertest parsertest)

add_executable(clcc-tests main.cpp ParserTest.cpp  ${BISON_parsertest_OUTPUTS} ${FLEX_lexertest_OUTPUTS})

add_dependencies(clcc-tests lexer parser)
target_link_libraries(clcc-tests ${GTEST_LIBRARIES})
