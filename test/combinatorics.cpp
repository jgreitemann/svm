/*   Support Vector Machine Library Wrappers
 *   Copyright (C) 2018-2019  Jonas Greitemann
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program, see the file entitled "LICENCE" in the
 *   repository's root directory, or see <http://www.gnu.org/licenses/>.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

#include <utility>
#include <vector>

#include <combinatorics/combinatorics.hpp>

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
