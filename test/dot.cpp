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

#include <vector>

#include <svm/dataset.hpp>


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
