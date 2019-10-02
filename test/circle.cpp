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
#include "model_test.hpp"
#include "circle_model.hpp"

#include <random>

#include <svm/kernel/polynomial.hpp>
#include <svm/kernel/rbf.hpp>
#include <svm/kernel/sigmoid.hpp>


TEST_CASE("circle-poly2-center") {
    std::mt19937 rng(42);
    circle_model trial_model({0., 0.}, 0.7, 2);
    model_test<svm::kernel::polynomial<2>>(1000, 0.98, trial_model, rng, 0.01);
}

TEST_CASE("circle-poly2-offset") {
    std::mt19937 rng(42);
    circle_model trial_model({0.3, 0.2}, 0.3);
    model_test<svm::kernel::polynomial<2>>(1000, 0.9, trial_model, rng, 0.45);
}

TEST_CASE("circle-rbf") {
    std::mt19937 rng(42);
    circle_model trial_model({0.3, 0.2}, 0.3);
    model_test<svm::kernel::rbf>(1000, 0.98, trial_model, rng, 0.01);
}

TEST_CASE("circle-sigmoid") {
    std::mt19937 rng(42);
    circle_model trial_model({0.3, 0.2}, 0.3);
    model_test<svm::kernel::sigmoid>(1000, 0.8, trial_model, rng, 0.45);
}
