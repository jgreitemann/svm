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
#include "hyperplane_model.hpp"
#include "model_test.hpp"

#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include <svm/kernel/linear.hpp>
#include <svm/kernel/linear_precomputed.hpp>
#include <svm/model.hpp>
#include <svm/parameters.hpp>
#include <svm/problem.hpp>


template <class Kernel>
void hyperplane_coeffs_test (size_t N, size_t M, double eps) {
    std::mt19937 rng(42);

    hyperplane_model trial_model(N, rng);

    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(M, rng, trial_model),
        params);
    using input_t = typename svm::model<Kernel>::input_container_type;
    auto empirical_classifier = empirical_model.classifier(1., -1.);
    auto introspector = linear_introspect(empirical_classifier);

    std::vector<double> empirical_C(N);
    for (size_t i = 0; i < N; ++i)
        empirical_C[i] = introspector.coefficient(i);
    double norm_trial = 0, norm_emp = 0;
    for (double c : trial_model.coefficients())
        norm_trial += c * c;
    for (double c : empirical_C)
        norm_emp += c * c;

    // compare decision function values from model with those
    // manually calculated from the inferred hyperplane coeffs
    std::uniform_real_distribution<double> uniform;
    double d_calc, d_pred;
    for (size_t m = 0; m < N; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        d_calc = 0;
        auto itC = empirical_C.begin();
        for (double x : xs) {
            d_calc += *itC * x;
            ++itC;
        }
        d_calc -= introspector.right_hand_side();
        std::tie(std::ignore, d_pred) = empirical_classifier(input_t(std::move(xs)));
        CHECK(d_calc == doctest::Approx(d_pred));
    }

    // compare empirical coeffs with input trial coeffs
    for (double & c : empirical_C)
        c *= sqrt(norm_trial / norm_emp);
    auto it = trial_model.coefficients().begin();
    for (size_t n = 0; n < N; ++n, ++it) {
        CHECK(*it == doctest::Approx(empirical_C[n]).epsilon(eps));
    }
}

TEST_CASE("hyperplane-coeffs-builtin") {
    hyperplane_coeffs_test<svm::kernel::linear>(4, 25000, 0.025);
}

TEST_CASE("hyperplane-coeffs-precomputed") {
    hyperplane_coeffs_test<svm::kernel::linear_precomputed>(4, 5000, 0.1);
}
