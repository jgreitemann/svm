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

#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <svm/model.hpp>
#include <svm/parameters.hpp>
#include <svm/problem.hpp>


template <class Problem, class TrialModel, class RNG_t>
Problem fill_problem (size_t M, RNG_t & rng,
                      TrialModel const& trial_model)
{
    std::uniform_real_distribution<double> uniform;
    Problem prob(trial_model.dim());
    using input_t = typename Problem::input_container_type;

    int ones = 0;
    double y;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(trial_model.dim());
        for (double & x : xs)
            x = uniform(rng);
        std::tie(y, std::ignore) = trial_model(xs);
        if (y > 0)
            ++ones;
        prob.add_sample(input_t(std::move(xs)), y);
    }
    std::cout << "fraction of ones: " << 1. * ones / M << std::endl;
    return prob;
}

template <class ModelA, class ModelB, class RNG>
double test_model (size_t M, RNG & rng,
                   ModelA & model_a,
                   ModelB & model_b)
{
    std::uniform_real_distribution<double> uniform;
    using input_a_t = typename ModelA::input_container_type;
    using input_b_t = typename ModelB::input_container_type;

    CHECK(model_a.dim() == model_b.dim());
    size_t N = model_a.dim();

    int correct = 0;
    double y_a, y_b;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        std::tie(y_a, std::ignore) = model_a(input_a_t(xs));
        std::tie(y_b, std::ignore) = model_b(input_b_t(std::move(xs)));
        if (y_a * y_b > 0)
            ++correct;
    }
    return 1. * correct / M;
}

template <class Kernel, class TrialModel, class RNG_t = std::mt19937>
void model_test (size_t M, double threshold, TrialModel const& trial_model, RNG_t rng = RNG_t(42), double nu = 0.1) {
    using model_t = svm::model<Kernel>;
    svm::parameters<Kernel> params(nu);

    model_t empirical_model(
        fill_problem<svm::problem<Kernel>>(M, rng, trial_model),
        params);

    size_t nr_labels = empirical_model.nr_labels();
    CHECK(nr_labels == 2);

    double success_rate = test_model(M, rng, trial_model, empirical_model);
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);
}
