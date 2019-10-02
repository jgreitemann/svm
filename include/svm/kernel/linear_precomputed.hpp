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

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <svm/dataset.hpp>
#include <svm/problem.hpp>
#include <svm/parameters.hpp>

#include <svm/detail/basic_parameters.hpp>

#include <svm/libsvm/svm.h>


namespace svm {

    namespace kernel {

        struct linear_precomputed {
            typedef std::vector<double> input_container_type;
            double operator() (input_container_type const& xi,
                               input_container_type const& xj) const {
                auto iti = xi.begin();
                auto itj = xj.begin();
                double sum = 0;
                for (; iti != xi.end() && itj != xj.end(); ++iti, ++itj) {
                    sum += *iti * *itj;
                }
                return sum;
            }
        };

    }

    template <>
    class parameters<kernel::linear_precomputed> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != PRECOMPUTED) {
                std::invalid_argument("parameters do not use precomputed kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = PRECOMPUTED;
        }
    };

}
