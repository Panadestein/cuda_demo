# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.8)

# Define the project name and the languages used (C++, Fortran)
project(cuda_fortran_mwe CXX Fortran)

# Enable CUDA language for this project
enable_language(CUDA)

# Find the CUDA library and include its directories
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find the BLAS library
find_package(BLAS REQUIRED)

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3")

# Set compiler flags for debugging and warnings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -g -Wall")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -Xcompiler -Wall")
endif()

# Add an executable target called 'cufor_isoc' built from 'mwe.f90' and 'loops.cu'
add_executable(cufor_isoc mwe.f90 loops.cu)

# Link the BLAS libraries, CUDA libraries, and the cuBLAS library (find_package sets this variables)
target_link_libraries(cufor_isoc ${BLAS_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})