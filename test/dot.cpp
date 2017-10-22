#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "dataset.hpp"

#include <vector>


TEST_CASE("dot") {
    std::vector<double> a = {3,  1, 4, 1, 5, 0, -9, 2};
    std::vector<double> b = {1, -1, 0, 1, 0, 0,  1};

    auto d_a = svm::dataset(a);
    auto d_b = svm::dataset(b);

    CHECK(d_a.dot(d_a) == doctest::Approx(137.));
    CHECK(d_b.dot(d_b) == doctest::Approx(4.));
    CHECK(d_a.dot(d_b) == doctest::Approx(-6.));
    CHECK(d_b.dot(d_a) == doctest::Approx(-6.));
}
