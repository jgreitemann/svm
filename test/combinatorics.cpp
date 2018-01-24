#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "combinatorics.hpp"

#include <vector>

using namespace combinatorics;


TEST_CASE("pascal-triangle") {
    std::vector<int> bc = {1, 1};
    for (size_t n = 2; n < 12; ++n) {
        std::vector<int> nbc;
        nbc.push_back(1);
        for (size_t p = 1; p < n; ++p) {
            nbc.push_back(bc[p-1] + bc[p]);
        }
        nbc.push_back(1);
        for (size_t k = 0; k <= n; ++k) {
            CHECK(binomial(n, k) == nbc[k]);
        }
        bc = std::move(nbc);
    }
}


TEST_CASE("multinomial") {
    CHECK(multinomial({2, 2, 2}) == 90);
    CHECK(multinomial({0, 1, 1, 4}) == 30);
    CHECK(multinomial({1, 0, 1, 4}) == 30);
    CHECK(multinomial({1, 2, 2, 3}) == 1680);
    CHECK(multinomial({1, 3, 3, 3}) == 16800);
}


TEST_CASE("binomial-from-multinomial") {
    for (size_t n = 2; n < 12; ++n) {
        for (size_t k = 0; k <= n; ++k) {
            CHECK(binomial(n, k) == multinomial({k, n-k}));
        }
    }
}


TEST_CASE("number-of-permutations") {
    CHECK(number_of_permutations(std::vector<int> {1, 4, 1, 4, 2, 1, 3, 5}) == 3360);
    CHECK(number_of_permutations(std::vector<int> {3, 1, 4, 1, 5, 9, 2, 6}) == 20160);
    CHECK(number_of_permutations(std::vector<int> {1, 1, 1, 1, 1}) == 1);
    CHECK(number_of_permutations(std::vector<int> {1, 1, 2, 1, 1}) == 5);
    CHECK(number_of_permutations(std::vector<int> {}) == 1);
}
