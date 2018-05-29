cmake_minimum_required(VERSION 2.8.12)
project(svm CXX)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(libsvm)

include_directories(.)
target_include_directories(svm PUBLIC .)

option(SVM__ENABLE_TESTS "If tests should be compiled" ON)
if(SVM__ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
