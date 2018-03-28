/*   Support Vector Machine Library Wrappers
 *   Copyright (C) 2018  Jonas Greitemann
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

#include "doctest.h"
#include "svm-wrapper.hpp"

#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <tuple>


using svm::detail::basic_problem;

template <class Container, class Label>
void test_problems_equal(basic_problem<Container, Label> const& lhs,
                         basic_problem<Container, Label> const& rhs)
{
    CHECK(lhs.dim() == rhs.dim());
    CHECK(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        Container xl, xr;
        Label yl, yr;
        std::tie(xl, yl) = lhs[i];
        std::tie(xr, yr) = lhs[i];
        CHECK(xl.size() == lhs.dim());
        CHECK(xr.size() == rhs.dim());
        auto it_l = xl.begin();
        auto it_r = xr.begin();
        for (; it_l < xl.end(); ++it_l, ++it_r)
            CHECK(*it_l == *it_r);
        CHECK(yl == yr);
    }
}

TEST_CASE("problem-append") {
    using C = std::vector<int>;
    using prob = basic_problem<C, int>;

    prob a(3), b(3), c(3);

    int i = 0;
    for (i = 0; i < 21; ++i) {
        a.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
        b.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
    }
    for (; i < 42; ++i) {
        a.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
        c.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
    }

    b.append_problem(std::move(c));

    test_problems_equal(a, b);
    CHECK(c.size() == 0);
}

TEST_CASE("problem-map-construct") {
    using C = std::vector<int>;
    using prob = basic_problem<C, int>;

    prob a(3), c(3);

    for (int i = 0; i < 42; ++i) {
        a.add_sample(C {3*i, 3*i+1, 3*i+2}, i % 3);
        c.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
    }

    prob b(std::move(c), [] (int i) { return i; });

    test_problems_equal(a, b);
    CHECK(c.size() == 0);
}

TEST_CASE("problem-map-append") {
    using C = std::vector<int>;
    using prob = basic_problem<C, int>;

    prob a(3), b(3), c(3);

    int i = 0;
    for (i = 0; i < 21; ++i) {
        a.add_sample(C {3*i, 3*i+1, 3*i+2}, i % 3);
        b.add_sample(C {3*i, 3*i+1, 3*i+2}, i % 3);
    }
    for (; i < 42; ++i) {
        a.add_sample(C {3*i, 3*i+1, 3*i+2}, i % 3);
        c.add_sample(C {3*i, 3*i+1, 3*i+2}, i);
    }

    b.append_problem(std::move(c), [] (int i) { return i; });

    test_problems_equal(a, b);
    CHECK(c.size() == 0);
}

TEST_CASE("problem-map-binary-classification") {
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, cmplx>;
    using C = typename problem_t::input_container_type;

    const size_t M = 1000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        prob.add_sample(C {c.real(), c.imag()}, c);
    }

    auto classifier = [] (cmplx c) -> double {
        double angle = std::arg(c);
        return (angle > 1. || angle < 1.-M_PI) ? +1. : -1.;
    };

    svm::problem<kernel_t, double> mapped_problem(std::move(prob), classifier);

    svm::model<kernel_t> model(std::move(mapped_problem), svm::parameters<kernel_t> {});
    double succ = 0.;
    double label, dec;
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        std::tie(label, dec) = model(C {c.real(), c.imag()});
        if (classifier(c) == doctest::Approx(label))
            succ += 1.;
    }
    succ /= M;
    std::cout << "success rate: " << succ << std::endl;
    CHECK(succ > 0.99);
}
