include_directories(..)
include_directories(../libsvm)
include_directories(doctest)

add_executable(hyperplane hyperplane.cpp ../data_view.cpp)
target_link_libraries(hyperplane svm)
add_test(hyperplane hyperplane)

add_executable(dot dot.cpp ../data_view.cpp)
add_test(dot dot)

add_executable(data_view data_view.cpp ../data_view.cpp)
add_test(data_view data_view)
