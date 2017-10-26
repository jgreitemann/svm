#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "hyperplane_test.hpp"


TEST_CASE("hyperplane-builtin") {
    hyperplane_test<svm::kernel::linear>(25, 2500, 0.98);
}
