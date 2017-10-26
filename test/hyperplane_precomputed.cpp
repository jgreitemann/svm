#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "hyperplane_test.hpp"
#include "kernel/linear_precomputed.hpp"


TEST_CASE("hyperplane-precomputed") {
    hyperplane_test<svm::kernel::linear_precomputed>(25, 2500, 0.98);
}

