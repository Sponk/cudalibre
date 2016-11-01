find_package(OpenCL REQUIRED)

if(USE_CL1)
  add_definitions(-DUSE_CL1)
endif()

add_library(lcruntime SHARED librecuda.cpp librecuda.h)
target_link_libraries(lcruntime ${OpenCL_LIBRARIES})
