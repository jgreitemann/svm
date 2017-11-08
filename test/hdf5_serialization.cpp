#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "serialization_test.hpp"
#include "kernel/linear_precomputed.hpp"
#include "hdf5_serialization.hpp"


TEST_CASE("serializer-hdf5-builtin") {
    serializer_test<svm::kernel::linear, svm::hdf5_tag>(25, 2500, 0.98, "hdf5-builtin.h5");
}

TEST_CASE("serializer-hdf5-precomputed") {
    serializer_test<svm::kernel::linear_precomputed, svm::hdf5_tag>(25, 2500, 0.98, "hdf5-precomputed.h5");
}
