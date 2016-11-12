
macro(cuda_add_executable name)
    set(_SOURCELIST "")
    set(_CUDA_DEPS "")
    foreach(arg ${ARGN})

        string(FIND ${arg} ".cu" _CU_POS REVERSE)
        string(LENGTH ${arg} _CU_LEN)

        if(${_CU_POS} STREQUAL -1)
            list(APPEND _SOURCELIST ${arg})
        else()
            math(EXPR _CU_POS "(${_CU_LEN})-(${_CU_POS})")
            if(${_CU_POS} STREQUAL 3)
                add_custom_target(${arg}_clcc ALL COMMAND ${CLCC_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/${arg} -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)
                execute_process(COMMAND ${CLCC_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/${arg} -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)

                list(APPEND _SOURCELIST ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)
                list(APPEND _CUDA_DEPS ${arg}_clcc)
            endif()
        endif(${_CU_POS} STREQUAL -1)
    endforeach()

    add_executable(${name} ${_SOURCELIST})
    add_dependencies(${name} ${_CUDA_DEPS})
    target_link_libraries(${name} ${LibreCuda_LIBRARIES} ${OpenCL_LIBRARIES})
    target_include_directories(${name} PUBLIC ${LibreCuda_INCLUDES})
endmacro()
