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

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>


using svm::detail::basic_problem;

SVM_LABEL_BEGIN(quadrants, 4)
SVM_LABEL_ADD(NORTH_EAST)
SVM_LABEL_ADD(NORTH_WEST)
SVM_LABEL_ADD(SOUTH_WEST)
SVM_LABEL_ADD(SOUTH_EAST)
SVM_LABEL_END()

TEST_CASE("4-class-classifiers") {
    using label_t = quadrants::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, label_t>;
    using C = typename problem_t::input_container_type;

    const size_t M = 10000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        label_t l = [c] {
            double angle = std::arg(c);
            if (angle < -M_PI/2) return quadrants::SOUTH_WEST;
            if (angle < 0) return quadrants::SOUTH_EAST;
            if (angle <  M_PI/2) return quadrants::NORTH_EAST;
            return quadrants::NORTH_WEST;
        } ();
        prob.add_sample(C {c.real(), c.imag()}, l);
    }

    using model_t = svm::model<kernel_t, label_t>;
    size_t nr_labels = model_t::nr_labels;
    size_t nr_classifiers = model_t::nr_classifiers;

    model_t model(std::move(prob), svm::parameters<kernel_t> {0.01});

    std::vector<label_t> all_labels = {quadrants::NORTH_WEST,
                                       quadrants::SOUTH_EAST,
                                       quadrants::NORTH_EAST,
                                       quadrants::SOUTH_WEST};
    std::sort(all_labels.begin(), all_labels.end());

    auto labels = model.labels();
    std::equal(all_labels.begin(), all_labels.end(), labels.begin());

    auto classifiers = model.classifiers();
    auto check_consistency = [&] (size_t c, size_t i, size_t j) {
        CHECK(classifiers[c].labels().first  == labels[i]);
        CHECK(classifiers[c].labels().second == labels[j]);
        auto clss = model.classifier(labels[i], labels[j]);
        CHECK(clss.labels().first  == labels[i]);
        CHECK(clss.labels().second == labels[j]);
    };

    check_consistency(0, 0, 1);
    check_consistency(1, 0, 2);
    check_consistency(2, 0, 3);
    check_consistency(3, 1, 2);
    check_consistency(4, 1, 3);
    check_consistency(5, 2, 3);

    for (size_t i = 0; i < 25; ++i) {
        C c {uniform(rng), uniform(rng)};
        auto res = model(c);
        for (size_t i = 0; i < nr_classifiers; ++i) {
            auto cres = classifiers[i](c);
            CHECK(res.first == cres.first);
            CHECK(res.second[i] == cres.second);
        }
    }
}
