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
#include "hyperplane_model.hpp"
#include "model_test.hpp"
#include "test_problems_equal.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <svm/model.hpp>
#include <svm/parameters.hpp>
#include <svm/problem.hpp>
#include <svm/serialization/serializer.hpp>


template <class Kernel, class Tag>
void model_serializer_test (size_t N, size_t M, double threshold, std::string const& name) {
    std::mt19937 rng(42);

    hyperplane_model trial_model(N, rng);
    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(M, rng, trial_model),
        params);

    svm::serialization::model_serializer<Tag, svm::model<Kernel>> saver(empirical_model);
    saver.save(name);

    svm::model<Kernel> restored_model;
    svm::serialization::model_serializer<Tag, svm::model<Kernel>> loader(restored_model);
    loader.load(name);

    double success_rate = test_model(M, rng, trial_model, restored_model);
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);

    success_rate = test_model(M, rng, empirical_model, restored_model);
    CHECK(success_rate == doctest::Approx(1.));
}

struct custom_label {
    static const size_t label_dim = 2;
    custom_label(double x, double y) { xs[0] = x; xs[1] = y; }
    template <class Iterator>
    custom_label(Iterator begin) { xs[0] = *begin; xs[1] = *(begin+1); }
    double const * begin() const { return xs; }
    double const * end() const { return xs + 2; }
    friend bool operator== (custom_label lhs, custom_label rhs) {
        return lhs.xs[0] == doctest::Approx(rhs.xs[0])
            && lhs.xs[1] == doctest::Approx(rhs.xs[1]);
    }
private:
    double xs[2];
};

template <class Kernel, class Tag>
void problem_serializer_test (size_t N, size_t M, std::string const& name) {
    using mapped_problem_t = svm::problem<Kernel, custom_label>;
    std::mt19937 rng(42);

    hyperplane_model trial_model(N, rng);
    auto orig_prob = fill_problem<svm::problem<Kernel>>(M, rng, trial_model);

    auto label_map = [] (double l) -> custom_label { return { l, (l+2)*(l+2) }; };
    auto mapped_prob = mapped_problem_t(std::move(orig_prob), label_map);

    svm::serialization::problem_serializer<Tag, mapped_problem_t> saver(mapped_prob);
    saver.save(name);

    mapped_problem_t restored_prob(0);
    svm::serialization::problem_serializer<Tag, mapped_problem_t> loader(restored_prob);
    loader.load(name);

    test_problems_equal(mapped_prob, restored_prob);
}
