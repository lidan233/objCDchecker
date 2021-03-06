cmake_minimum_required(VERSION 3.17)
project(objViewer LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")


find_package(CUDA  REQUIRED)
include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\include")


include(cmake/LidanBase.cmake)
include(cmake/LidanBuild.cmake)

ADD_SUBSRC(incdir inc)
ADD_SUBSRC(srcdir src)
ADD_SUBSRC(bvhdir BVH)
ADD_SUBSRC(cudalibdir cudalib)

add_subdirectory(external/cxxopts)

include_directories(inc cudalib src BVH external/cxxopts/include)
link_directories(lib )

add_executable(objViewer  ${incdir} ${srcdir} ${cudalibdir} ${bvhdir} )
target_link_libraries(objViewer glut64)