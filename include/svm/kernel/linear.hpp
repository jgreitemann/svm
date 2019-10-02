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
#include <utility>

#include <svm/model.hpp>
#include <svm/problem.hpp>
#include <svm/parameters.hpp>

#include <svm/detail/basic_parameters.hpp>
#include <svm/detail/patch_through_problem.hpp>

#include <svm/libsvm/svm.h>


namespace svm {

    namespace kernel {

        struct linear {
        };

    }

    template <class Label>
    class problem<kernel::linear, Label> : public detail::patch_through_problem<Label> {
        using detail::patch_through_problem<Label>::patch_through_problem;
    };

    template <>
    class parameters<kernel::linear> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != LINEAR) {
                std::invalid_argument("parameters do not use linear kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = LINEAR;
        }
    };

    template <class Classifier>
    struct linear_introspector {

        linear_introspector(Classifier const& cl) : classifier(cl) {}

        double coefficient(size_t i) const {
            double c = 0;
            for (auto p : classifier) {
                double yalpha = p.first;
                auto const& x = p.second;
                auto itX = x.begin();
                std::advance(itX, i);
                c += yalpha * *itX;
            }
            return c;
        }

        double right_hand_side() const {
            return classifier.rho();
        }

    private:
        Classifier classifier;
    };

    template <class Classifier>
    linear_introspector<Classifier> linear_introspect (Classifier const& cl) {
        return linear_introspector<Classifier> {cl};
    }

}
