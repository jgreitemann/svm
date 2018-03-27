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

#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <algorithm>
#include <stdexcept>
#include <utility>


namespace svm {

    namespace kernel {

        struct linear;

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

    template <class Kernel>
    struct linear_introspector {
        using model_t = model<Kernel>;

        linear_introspector(model_t const& model) : model_(model) {}
        
        double coefficient(size_t i) const {
            double c = 0;
            for (auto p : model_) {
                double yalpha = p.first;
                auto const& x = p.second;
                auto itX = x.begin();
                std::advance(itX, i);
                c += yalpha * *itX;
            }
            return c;
        }

    private:
        model_t const& model_;
    };

}
