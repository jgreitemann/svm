include_directories(../include)

add_executable(combinatorics combinatorics.cpp)
add_test(combinatorics combinatorics)

add_executable(problem problem.cpp)
target_link_libraries(problem svm)
add_test(problem problem)

add_executable(hyperplane-builtin hyperplane_builtin.cpp)
target_link_libraries(hyperplane-builtin svm)
add_test(hyperplane-builtin hyperplane-builtin)

add_executable(hyperplane-precomputed hyperplane_precomputed.cpp)
target_link_libraries(hyperplane-precomputed svm)
add_test(hyperplane-precomputed hyperplane-precomputed)

add_executable(hyperplane-poly hyperplane_poly.cpp)
target_link_libraries(hyperplane-poly svm)
add_test(hyperplane-poly hyperplane-poly)

add_executable(hyperplane-sigmoid hyperplane_sigmoid.cpp)
target_link_libraries(hyperplane-sigmoid svm)
add_test(hyperplane-sigmoid hyperplane-sigmoid)

add_executable(hyperplane-coeffs hyperplane_coeffs.cpp)
target_link_libraries(hyperplane-coeffs svm)
add_test(hyperplane-coeffs hyperplane-coeffs)

add_executable(circle circle.cpp)
target_link_libraries(circle svm)
add_test(circle circle)

add_executable(dot dot.cpp)
target_link_libraries(dot svm)
add_test(dot dot)

add_executable(data_view data_view.cpp)
target_link_libraries(data_view svm)
add_test(data_view data_view)

add_executable(poly-introspect poly_introspect.cpp)
target_link_libraries(poly-introspect svm)
add_test(poly-introspect poly-introspect)

add_executable(multi-introspect multi_introspect.cpp)
target_link_libraries(multi-introspect svm)
add_test(multi-introspect multi-introspect)

add_executable(label-classifier-stability label_classifier_stability.cpp)
target_link_libraries(label-classifier-stability svm)
add_test(label-classifier-stability label-classifier-stability)

add_executable(ascii-serialization ascii_serialization.cpp)
target_link_libraries(ascii-serialization svm)
add_test(ascii-serialization ascii-serialization)

find_package(ALPSCore COMPONENTS hdf5)
if (ALPSCore_LIBRARIES)
  add_executable(hdf5-serialization hdf5_serialization.cpp)
  target_link_libraries(hdf5-serialization svm ${ALPSCore_LIBRARIES})
  add_test(hdf5-serialization hdf5-serialization)
else()
  message(WARNING "ALPSCore not found; skipping HDF5 tests")
endif()
