#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "dataset.hpp"

#include <iostream>
#include <vector>

void test_data_view (std::vector<double> const& a) {
    std::vector<double> b;
    svm::dataset d(a);
    for (double x : d.view()) {
        b.push_back(x);
        std::cout << x << std::endl;
    }
    CHECK(a.size() == b.size());
    CHECK(a == b);
}

TEST_CASE("dataset-view") {
    test_data_view({3, 1, 4, 1, 5, 0, -9, 2});
    test_data_view({0, 1, 4, 1, 5, 0, -9, 2});
}
