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
#include "serialization_test.hpp"

#include <svm/kernel/linear.hpp>
#include <svm/kernel/linear_precomputed.hpp>
#include <svm/serialization/hdf5.hpp>


TEST_CASE("model-serializer-hdf5-builtin") {
    model_serializer_test<svm::kernel::linear, svm::hdf5_tag>(4, 1000, 0.99, "hdf5-builtin-model.h5");
}

TEST_CASE("model-serializer-hdf5-precomputed") {
    model_serializer_test<svm::kernel::linear_precomputed, svm::hdf5_tag>(4, 1000, 0.99, "hdf5-precomputed-model.h5");
}

TEST_CASE("problem-serializer-hdf5-builtin") {
    problem_serializer_test<svm::kernel::linear, svm::hdf5_tag>(4, 1000, "hdf5-builtin-problem.h5");
}

TEST_CASE("problem-serializer-hdf5-precomputed") {
    problem_serializer_test<svm::kernel::linear_precomputed, svm::hdf5_tag>(4, 1000, "hdf5-precomputed-problem.h5");
}
