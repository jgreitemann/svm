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

#pragma once

#include "doctest/doctest.h"

#include <svm/detail/basic_problem.hpp>


using svm::detail::basic_problem;

template <class Container, class Label>
void test_problems_equal(basic_problem<Container, Label> const& lhs,
                         basic_problem<Container, Label> const& rhs)
{
    CHECK(lhs.dim() == rhs.dim());
    CHECK(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        Container const& xl = lhs[i].first, xr = rhs[i].first;
        Label yl = lhs[i].second, yr = rhs[i].second;
        auto it_l = xl.begin();
        auto it_r = xr.begin();
        for (size_t j = 0; j < lhs.dim(); ++j, ++it_l, ++it_r)
            CHECK(*it_l == doctest::Approx(*it_r));
        CHECK(it_l == xl.end());
        CHECK(it_r == xr.end());
        CHECK(yl == yr);
    }
}
