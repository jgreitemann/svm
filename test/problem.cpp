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
#include "label.hpp"
#include "test_problems_equal.hpp"

#include <complex>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>


using svm::detail::basic_problem;

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

    prob b(std::move(c), [] (int i) { return i % 3; });

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

    b.append_problem(std::move(c), [] (int i) { return i % 3; });

    test_problems_equal(a, b);
    CHECK(c.size() == 0);
}

SVM_LABEL_BEGIN(binary_class, 2)
SVM_LABEL_ADD(WHITE)
SVM_LABEL_ADD(BLACK)
SVM_LABEL_END()

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

    auto classifier = [] (cmplx c) -> binary_class::label {
        double angle = std::arg(c);
        return (angle > 1. || angle < 1.-M_PI) ? binary_class::WHITE : binary_class::BLACK;
    };

    svm::problem<kernel_t, binary_class::label> mapped_problem(std::move(prob), classifier);

    using model_t = svm::model<kernel_t, binary_class::label>;
    size_t nr_labels = model_t::nr_labels;
    size_t nr_classifiers = model_t::nr_classifiers;
    CHECK(nr_labels == 2);
    CHECK(nr_classifiers == 1);
    static_assert(std::is_same<typename model_t::decision_type, double>::value,
                  "wrong decision type for binary classification");
    model_t model(std::move(mapped_problem), svm::parameters<kernel_t> {});
    double succ = 0.;
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        auto label = model(C {c.real(), c.imag()}).first;
        if (classifier(c) == label)
            succ += 1.;
    }
    succ /= M;
    std::cout << "success rate: " << succ << std::endl;
    CHECK(succ > 0.99);
}


SVM_LABEL_BEGIN(ternary_class, 3)
SVM_LABEL_ADD(RED)
SVM_LABEL_ADD(GREEN)
SVM_LABEL_ADD(BLUE)
SVM_LABEL_END()

TEST_CASE("problem-map-ternary-classification") {
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

    auto classifier = [] (cmplx c) -> ternary_class::label {
        double angle = std::arg(c);
        if (angle < -1) return ternary_class::RED;
        if (angle <  1) return ternary_class::GREEN;
        return ternary_class::BLUE;
    };

    svm::problem<kernel_t, ternary_class::label> mapped_problem(std::move(prob), classifier);

    using model_t = svm::model<kernel_t, ternary_class::label>;
    size_t nr_labels = model_t::nr_labels;
    size_t nr_classifiers = model_t::nr_classifiers;
    CHECK(nr_labels == 3);
    CHECK(nr_classifiers == 3);
    static_assert(std::is_same<typename model_t::decision_type, std::array<double,3>>::value,
                  "wrong decision type for ternary classification");

    model_t model(std::move(mapped_problem), svm::parameters<kernel_t> {});
    double succ = 0.;
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        auto res = model(C {c.real(), c.imag()});
        auto label = res.first;
        auto dec = res.second;
        CHECK(std::distance(dec.begin(), dec.end()) == nr_classifiers);
        if (classifier(c) == label)
            succ += 1.;
    }
    succ /= M;
    std::cout << "success rate: " << succ << std::endl;
    CHECK(succ > 0.99);
}
