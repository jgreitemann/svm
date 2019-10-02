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
#include "test_problems_equal.hpp"

#include <complex>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <svm/dataset.hpp>
#include <svm/label.hpp>
#include <svm/model.hpp>
#include <svm/parameters.hpp>
#include <svm/problem.hpp>
#include <svm/detail/basic_problem.hpp>
#include <svm/kernel/linear.hpp>


using svm::detail::basic_problem;

SVM_LABEL_BEGIN(binary_class, 2)
SVM_LABEL_ADD(RED)
SVM_LABEL_ADD(BLUE)
SVM_LABEL_END()

SVM_LABEL_BEGIN(ternary_class, 3)
SVM_LABEL_ADD(RED)
SVM_LABEL_ADD(GREEN)
SVM_LABEL_ADD(BLUE)
SVM_LABEL_END()

TEST_CASE("ternary-introspection") {
    using label_t = ternary_class::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, label_t>;
    using C = typename problem_t::input_container_type;

    const size_t M = 10000;
    const double eps = 0.01;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        label_t l = [c] {
            double angle = std::arg(c);
            if (angle < -1) return ternary_class::RED;
            if (angle <  1) return ternary_class::GREEN;
            return ternary_class::BLUE;
        } ();
        prob.add_sample(C {c.real(), c.imag()}, l);
    }

    using model_t = svm::model<kernel_t, label_t>;
    model_t model(std::move(prob), svm::parameters<kernel_t> {0.01});
    size_t nr_labels = model.nr_labels();
    CHECK(nr_labels == 3);
    size_t nr_classifiers = model.nr_classifiers();
    CHECK(nr_classifiers == 3);

    using introspector_t = svm::linear_introspector<model_t::classifier_type>;
    auto check_slope = [&] (label_t l1, label_t l2, double s) {
        auto classifier = model.classifier(l1, l2);
        introspector_t introspector(classifier);
        double slope = -introspector.coefficient(0) / introspector.coefficient(1);
        double intercept = classifier.rho() / introspector.coefficient(1);
        std::cout << l1 << '/' << l2 << ": slope " << slope
                  << ", y-intercept: " << intercept << std::endl;
        CHECK(slope == doctest::Approx(s).epsilon(eps));
        CHECK(intercept == doctest::Approx(0.).epsilon(eps));
    };

    check_slope(ternary_class::RED, ternary_class::GREEN, tan(-1));
    check_slope(ternary_class::BLUE, ternary_class::GREEN, tan(1));
    check_slope(ternary_class::RED, ternary_class::BLUE, 0.);
}

TEST_CASE("classifier-consistency") {
    using label2_t = binary_class::label;
    using label3_t = ternary_class::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem2_t = svm::problem<kernel_t, label2_t>;
    using problem3_t = svm::problem<kernel_t, label3_t>;
    using C = typename problem2_t::input_container_type;

    const size_t M = 1000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem2_t prob2(2);
    problem3_t prob3(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        double angle = std::arg(c);
        if (angle < -1) {
            prob2.add_sample(C {c.real(), c.imag()},  binary_class::RED);
            prob3.add_sample(C {c.real(), c.imag()}, ternary_class::RED);
        } else if (angle <  1) {
            prob3.add_sample(C {c.real(), c.imag()}, ternary_class::GREEN);
        } else {
            prob2.add_sample(C {c.real(), c.imag()},  binary_class::BLUE);
            prob3.add_sample(C {c.real(), c.imag()}, ternary_class::BLUE);
        }
    }

    using model2_t = svm::model<kernel_t, label2_t>;
    using model3_t = svm::model<kernel_t, label3_t>;

    model2_t model2(std::move(prob2), svm::parameters<kernel_t> {});
    model3_t model3(std::move(prob3), svm::parameters<kernel_t> {});

    auto cl2 = model2.classifier();
    auto cl3 = model3.classifier(ternary_class::RED, ternary_class::BLUE);
    auto cl3_rev = model3.classifier(ternary_class::BLUE, ternary_class::RED);
    CHECK(cl3.labels().first == ternary_class::RED);
    CHECK(cl3.labels().second == ternary_class::BLUE);
    CHECK(cl3_rev.labels().first == ternary_class::BLUE);
    CHECK(cl3_rev.labels().second == ternary_class::RED);

    CHECK(cl2.rho() == doctest::Approx(cl3.rho()));

    auto it2 = cl2.begin();
    auto it3 = cl3.begin();
    double ya2, ya3;
    svm::data_view sv2, sv3;
    while (it2 != cl2.end()) {
        std::tie(ya2, sv2) = *it2;
        std::tie(ya3, sv3) = *it3;
        if (ya3 == doctest::Approx(0.)) {
            ++it3;
            continue;
        }
        CHECK(ya2 == doctest::Approx(ya3));
        CHECK(sv2.front() == sv3.front());
        ++it2, ++it3;
    }
    while (it3 != cl3.end()) {
        std::tie(ya3, std::ignore) = *it3;
        CHECK(ya3 == doctest::Approx(0.));
        ++it3;
    }
}

TEST_CASE("introspector-consistency") {
    using label2_t = binary_class::label;
    using label3_t = ternary_class::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem2_t = svm::problem<kernel_t, label2_t>;
    using problem3_t = svm::problem<kernel_t, label3_t>;
    using C = typename problem2_t::input_container_type;

    const size_t N = 250;
    const size_t M = 1000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem2_t prob2(N);
    problem3_t prob3(N);
    for (size_t i = 0; i < M; ++i) {
        std::vector<double> v(N);
        for (auto & e : v)
            e = uniform(rng);
        cmplx c {v[0], v[1]};
        double angle = std::arg(c);
        if (angle < -1) {
            prob2.add_sample(C {v},  binary_class::RED);
            prob3.add_sample(C {v}, ternary_class::RED);
        } else if (angle <  1) {
            prob3.add_sample(C {v}, ternary_class::GREEN);
        } else {
            prob2.add_sample(C {v},  binary_class::BLUE);
            prob3.add_sample(C {v}, ternary_class::BLUE);
        }
    }

    using model2_t = svm::model<kernel_t, label2_t>;
    using model3_t = svm::model<kernel_t, label3_t>;

    model2_t model2(std::move(prob2), svm::parameters<kernel_t> {});
    model3_t model3(std::move(prob3), svm::parameters<kernel_t> {});

    auto cl2 = model2.classifier();
    auto cl3 = model3.classifier(ternary_class::RED, ternary_class::BLUE);

    auto introspector2 = svm::linear_introspect(cl2);
    auto introspector3 = svm::linear_introspect(cl3);

    for (size_t i = 0; i < N; ++i)
        CHECK(introspector2.coefficient(i)
              == doctest::Approx(introspector3.coefficient(i)));
}
