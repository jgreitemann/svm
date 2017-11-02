#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "hyperplane_test.hpp"


TEST_CASE("hyperplane-poly") {
    hyperplane_test<svm::kernel::polynomial<1>>(25, 2500, 0.98);
}
