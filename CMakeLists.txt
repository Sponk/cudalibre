cmake_minimum_required(VERSION 3.5)
project(CudaLibre)

# Version info
set(CUDALIBRE_MAJOR_VERSION 0)
set(CUDALIBRE_MINOR_VERSION 1)
set(CUDALIBRE_PATCH_VERSION 0)
set(CUDALIBRE_VERSION ${CUDALIBRE_MAJOR_VERSION}.${CUDALIBRE_MINOR_VERSION}.${CUDALIBRE_PATCH_VERSION})
set(CUDALIBRE_PACKAGE "CudaLibre")
SET(CUDALIBRE_VENDOR "Yannick Pflanzer")
SET(CUDALIBRE_COPYRIGHT "Copyright (c) 2016, ${CUDALIBRE_VENDOR}.")

set(INSTALL_LIB_DIR lib64 CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib64/CudaLibre)
endif()

set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

## Packaging
set(CPACK_PACKAGE_NAME "${CUDALIBRE_PACKAGE}")
SET(CPACK_PACKAGE_VENDOR "${CUDALIBRE_VENDOR}")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Source implementation of the CUDA compiler and runtime.")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
SET(CPACK_PACKAGE_VERSION_MAJOR "${CUDALIBRE_MAJOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_MINOR "${CUDALIBRE_MINOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_PATCH "${CUDALIBRE_PATCH_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CUDALIBRE_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")

SET(CPACK_PACKAGE_CONTACT Yannick Pflanzer <yp1995@live.de>)
SET(CPACK_GENERATOR "STGZ;TGZ;DEB;RPM;")
SET(CPACK_STRIP_FILES ON)

include(CPack)

set(CMAKE_MODULE_PATH
        ${CMAKE_SOURCE_DIR}/CMake
        ${CMAKE_SOURCE_DIR}/CMake/Packages
        ${CMAKE_MODULE_PATH}
        )

set(CMAKE_CXX_STANDARD 11)

set(CULCC_EXECUTABLE ${CMAKE_BINARY_DIR}/culcc/culcc)
set(CudaLibre_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/runtime ${CMAKE_SOURCE_DIR}/cudastd)
set(CudaLibre_LIBRARIES clruntime)
set(GTEST_LIBRARIES gtest)
set(GTEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/gtest/googletest/src)

include("CudaLibre")
enable_testing()

add_subdirectory(gtest)
add_subdirectory(culcc)
add_subdirectory(culccd)
add_subdirectory(runtime)
add_subdirectory(cudastd)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(docs)

# CMake install package
export(TARGETS culcc clruntime FILE "${PROJECT_BINARY_DIR}/CudaLibreTargets.cmake")
export(PACKAGE CudaLibre)

set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(CMake/CudaLibreConfig.cmake.in "${PROJECT_BINARY_DIR}/CudaLibreConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS "\${CUDALIBRE_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(CMake/CudaLibreConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CudaLibreConfig.cmake" @ONLY)

configure_file(CMake/CudaLibreConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/CudaLibreConfigVersion.cmake" @ONLY)

install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CudaLibreConfig.cmake"
        "${PROJECT_BINARY_DIR}/CudaLibreConfigVersion.cmake"
        DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT CudaLibreTargets DESTINATION
        "${INSTALL_CMAKE_DIR}" COMPONENT dev)

## Uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CMake/uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CMake/uninstall.cmake"
  IMMEDIATE @ONLY)

add_custom_target(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/CMake/uninstall.cmake")
