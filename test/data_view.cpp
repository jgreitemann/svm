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

#include <iostream>
#include <vector>

#include <svm/dataset.hpp>


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
