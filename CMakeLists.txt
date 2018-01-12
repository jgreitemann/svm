cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

enable_testing()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")

find_package(ALPSCore 1.0 REQUIRED)

add_subdirectory(libsvm)
add_subdirectory(test)
