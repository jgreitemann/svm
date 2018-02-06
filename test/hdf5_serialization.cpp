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

#include "serialization_test.hpp"
#include "kernel/linear_precomputed.hpp"
#include "hdf5_serialization.hpp"


TEST_CASE("serializer-hdf5-builtin") {
    serializer_test<svm::kernel::linear, svm::hdf5_tag>(25, 2500, 0.98, "hdf5-builtin.h5");
}

TEST_CASE("serializer-hdf5-precomputed") {
    serializer_test<svm::kernel::linear_precomputed, svm::hdf5_tag>(25, 2500, 0.98, "hdf5-precomputed.h5");
}
