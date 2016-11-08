
find_package(Doxygen REQUIRED)

set(DOC_INPUT ${CMAKE_SOURCE_DIR}/runtime)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doxy.in ${CMAKE_CURRENT_BINARY_DIR}/doxy.doxygen
)

add_custom_target(doc ALL
                    COMMAND ${DOXYGEN_EXECUTABLE}
                    ${CMAKE_CURRENT_BINARY_DIR}/doxy.doxygen
        )