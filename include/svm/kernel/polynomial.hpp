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
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <combinatorics/combinatorics.hpp>

#include <svm/model.hpp>
#include <svm/problem.hpp>
#include <svm/parameters.hpp>

#include <svm/detail/basic_parameters.hpp>
#include <svm/detail/patch_through_problem.hpp>

#include <svm/libsvm/svm.h>


namespace svm {

    namespace kernel {

        template <size_t D>
        struct polynomial {
            static size_t const Degree = D;
        };

    }

    template <size_t D, class Label>
    class problem<kernel::polynomial<D>, Label> : public detail::patch_through_problem<Label> {
        using detail::patch_through_problem<Label>::patch_through_problem;
    };

    template <size_t D>
    class parameters<kernel::polynomial<D>> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != POLY) {
                std::invalid_argument("parameters do not use polynomial kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = POLY;
            params.degree = D;
            params.gamma = 1.;
            params.coef0 = 0.;
        }
        double & gamma () { return params.gamma; }
        double const& gamma () const { return params.gamma; }
        double & coef0 () { return params.coef0; }
        double const& coef0 () const { return params.coef0; }
    };


    template <class Classifier, size_t K, size_t D = Classifier::kernel_type::Degree>
    struct tensor_introspector {
        static_assert(K <= D, "invalid tensor rank");

        tensor_introspector (Classifier const& cl)
            : classifier(cl)
        {
            using namespace combinatorics;
            fac = binomial(D, K) * ipow(classifier.params().gamma(), K)
                * ipow(classifier.params().coef0(), D-K);
        }

        template <size_t L=K, typename = typename std::enable_if<L != 0>::type>
        double tensor (std::array<size_t, K> ind) const {
            std::sort(ind.begin(), ind.end());
            double yalpha;
            data_view x;
            double sum = 0;
            for (auto p : classifier) {
                std::tie(yalpha, x) = std::move(p);
                double prod = 1.;
                size_t j = 0;
                auto itX = x.begin();
                for (size_t i : ind) {
                    std::advance(itX, i - j);
                    j = i;
                    prod *= *itX;
                }
                sum += yalpha * prod;
            }
            return fac * sum;
        }

        template <size_t L=K, typename = typename std::enable_if<L == 0>::type>
        double tensor () const {
            return fac - classifier.rho();
        }
    private:
        Classifier classifier;
        double fac;
    };

    template <size_t K, class Classifier>
    tensor_introspector<Classifier, K> tensor_introspect (Classifier const& cl) {
        return tensor_introspector<Classifier, K> {cl};
    }

}
