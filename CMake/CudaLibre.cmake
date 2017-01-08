## The cuda_add_executable macro for use in the CudaLibre build system only!
## It is being used to compile the regression tests
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
                file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp "") ## Ensure the file exists for use in the CudaLibre build system
                string(REPLACE "/" "_" _ARGNAME ${arg})

                add_custom_target(${arg}_culcc ALL COMMAND ${CULCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${arg}
                        -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp
                        -- -I${CMAKE_SOURCE_DIR}/ -I${CMAKE_SOURCE_DIR}/runtime -include math.cuh -I${CMAKE_CURRENT_SOURCE_DIR})

                # culcc is not yet built when configuring, so it can't be called yet.
                # execute_process(COMMAND ${CULCC_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/${arg} -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)

                add_dependencies(${arg}_culcc culcc)

                list(APPEND _SOURCELIST ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)
                list(APPEND _CUDA_DEPS ${arg}_culcc)
            endif()
        endif(${_CU_POS} STREQUAL -1)
    endforeach()

    add_executable(${name} ${_SOURCELIST})
    add_dependencies(${name} ${_CUDA_DEPS})
    target_link_libraries(${name} ${CudaLibre_LIBRARIES} ${OpenCL_LIBRARIES} cudastd)
    target_include_directories(${name} PRIVATE ${CudaLibre_INCLUDE_DIRS})
endmacro()

macro(cuda_add_library name)
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
                string(REPLACE "/" "_" _ARGNAME ${arg})
                file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp "") ## Ensure the file exists for use in the CudaLibre build system

                ## Get full source directory so we can search includes for '#include "some_local_include.h"'
                get_filename_component(_INCPATH ${CMAKE_CURRENT_SOURCE_DIR}/${arg} DIRECTORY)
                add_custom_target(${_ARGNAME}_culcc ALL COMMAND ${CULCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${arg}
                        -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp -- -I${_INCPATH})

                # culcc is not yet built when configuring, so it can't be called yet.
                # execute_process(COMMAND ${CULCC_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/${arg} -o ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)

                add_dependencies(${_ARGNAME}_culcc culcc)

                list(APPEND _SOURCELIST ${CMAKE_CURRENT_BINARY_DIR}/${arg}.cpp)
                list(APPEND _CUDA_DEPS ${_ARGNAME}_culcc)
            endif()
        endif(${_CU_POS} STREQUAL -1)
    endforeach()

    add_library(${name} SHARED ${_SOURCELIST})
    add_dependencies(${name} ${_CUDA_DEPS})
    target_link_libraries(${name} ${CudaLibre_LIBRARIES} ${OpenCL_LIBRARIES})
    target_include_directories(${name} PRIVATE ${CudaLibre_INCLUD_DIRS} ${_INCPATH})
endmacro()
