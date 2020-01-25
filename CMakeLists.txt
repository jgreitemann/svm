cmake_minimum_required(VERSION 2.8.12)
project(svm CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(svm STATIC src/svm.cpp)
target_include_directories(svm PUBLIC include)

option(SVM__ENABLE_TESTS "If tests should be compiled" ON)
if(SVM__ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

install(TARGETS svm PUBLIC_HEADER DESTINATION include)
install(DIRECTORY include/svm DESTINATION include)
install(DIRECTORY include/combinatorics DESTINATION include)
