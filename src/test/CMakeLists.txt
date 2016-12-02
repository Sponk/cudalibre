include_directories(
		${GTEST_INCLUDE_DIRS}
)

bison_target(parsertest ../parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
flex_target(lexertest ../lexer.l  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)
add_flex_bison_dependency(lexertest parsertest)

add_executable(clcc-tests main.cpp ../ClangTool.cpp ParserTest.cpp StructParserTest.cpp
		${BISON_parsertest_OUTPUTS} ${FLEX_lexertest_OUTPUTS})

target_link_libraries(clcc-tests
	${GTEST_LIBRARIES}
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

add_test(CompilerTest clcc-tests)
