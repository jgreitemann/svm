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

#include <svm/libsvm/svm.h>


namespace svm {

    enum class machine_type {
        C_SVC = C_SVC,
        NU_SVC = NU_SVC,
        ONE_CLASS = ONE_CLASS,
        EPSILON_SVR = EPSILON_SVR,
        NU_SVR = NU_SVR
    };

    namespace detail {

        class basic_parameters {
        public:
            basic_parameters (struct svm_parameter const& p) : params(p) {}

            basic_parameters (double reg = 0.1, machine_type mtype = machine_type::NU_SVC) {
                switch (mtype) {
                case machine_type::C_SVC:
                    params.C = reg;
                    break;
                case machine_type::NU_SVC:
                    params.nu = reg;
                    break;
                default:
                    throw std::logic_error("SVM type not supported");
                }
                params.svm_type = static_cast<int>(mtype);
                params.degree = 0;
                params.gamma = 0;
                params.coef0 = 0;
                params.cache_size = 100;
                params.eps = 1e-3;
                params.nr_weight = 0;
                params.weight_label = NULL;
                params.weight = NULL;
                params.shrinking = 1;
                params.probability = 0;
            }

            double cache_size() const { return params.cache_size; }
            double & cache_size() { return params.cache_size; }

            struct svm_parameter * svm_params_ptr () {
                return &params;
            }
        protected:
            struct svm_parameter params;
        };

    }

}
