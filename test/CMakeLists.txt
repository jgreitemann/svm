include_directories(..)
include_directories(../libsvm)
include_directories(doctest)

add_executable(hyperplane hyperplane.cpp)
target_link_libraries(hyperplane svm)
add_test(hyperplane hyperplane)

add_executable(dot dot.cpp)
add_test(dot dot)
