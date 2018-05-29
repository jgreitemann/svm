# include_directories(.)
add_library(svm STATIC svm.cpp)
target_include_directories(svm PUBLIC .)
