include_directories(..)
include_directories(../libsvm)
include_directories(doctest)

add_executable(hyperplane-builtin hyperplane_builtin.cpp ../data_view.cpp)
target_link_libraries(hyperplane-builtin svm)
add_test(hyperplane-builtin hyperplane-builtin)

add_executable(hyperplane-precomputed hyperplane_precomputed.cpp ../data_view.cpp)
target_link_libraries(hyperplane-precomputed svm)
add_test(hyperplane-precomputed hyperplane-precomputed)

add_executable(hyperplane-poly hyperplane_poly.cpp ../data_view.cpp)
target_link_libraries(hyperplane-poly svm)
add_test(hyperplane-poly hyperplane-poly)

add_executable(hyperplane-coeffs hyperplane_coeffs.cpp ../data_view.cpp)
target_link_libraries(hyperplane-coeffs svm)
add_test(hyperplane-coeffs hyperplane-coeffs)

add_executable(dot dot.cpp ../data_view.cpp)
add_test(dot dot)

add_executable(data_view data_view.cpp ../data_view.cpp)
add_test(data_view data_view)

add_executable(poly-introspect poly_introspect.cpp ../data_view.cpp)
target_link_libraries(poly-introspect svm)
add_test(poly-introspect poly-introspect)

add_executable(serialization serialization.cpp ../data_view.cpp)
target_link_libraries(serialization svm)
add_test(serialization serialization)
