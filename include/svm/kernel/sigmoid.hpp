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

#include <stdexcept>

#include <svm/problem.hpp>
#include <svm/parameters.hpp>
#include <svm/model.hpp>

#include <svm/detail/basic_parameters.hpp>
#include <svm/detail/patch_through_problem.hpp>

#include <svm/libsvm/svm.h>


namespace svm {

    namespace kernel {

        struct sigmoid {
        };

    }

    template <class Label>
    class problem<kernel::sigmoid, Label> : public detail::patch_through_problem<Label> {
        using detail::patch_through_problem<Label>::patch_through_problem;
    };

    template <>
    class parameters<kernel::sigmoid> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != SIGMOID) {
                std::invalid_argument("parameters do not use radial basis "
                                      "function (SIGMOID) kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = SIGMOID;
            params.gamma = 1.;
            params.coef0 = 0.;
        }
        double const& gamma () const { return params.gamma; }
        double & gamma () { return params.gamma; }
        double const& coef0 () const { return params.coef0; }
        double & coef0 () { return params.coef0; }
    };

}
