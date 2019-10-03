cmake_minimum_required(VERSION 2.8.12)
project(svm CXX)
set(CMAKE_CXX_STANDARD 14)

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
