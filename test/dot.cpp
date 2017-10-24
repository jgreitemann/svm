#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "dataset.hpp"

#include <vector>


TEST_CASE("dot") {
    std::vector<double> a = {3,  1, 4, 1, 5, 0, -9, 2};
    std::vector<double> b = {1, -1, 0, 1, 0, 0,  1};

    auto d_a = svm::dataset(a);
    auto d_b = svm::dataset(b);

    CHECK(dot(d_a, d_a) == doctest::Approx(137.));
    CHECK(dot(d_b, d_b) == doctest::Approx(4.));
    CHECK(dot(d_a, d_b) == doctest::Approx(-6.));
    CHECK(dot(d_b, d_a) == doctest::Approx(-6.));
}
