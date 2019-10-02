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

#include <type_traits>
#include <utility>
#include <vector>

#include <svm/dataset.hpp>
#include <svm/detail/always.hpp>
#include <svm/detail/basic_problem.hpp>
#include <svm/libsvm/svm.h>
#include <svm/traits/label_traits.hpp>


namespace svm {
    namespace detail {

        template <class Kernel, class Container, class Label>
        class precompute_kernel_problem : public basic_problem<Container, Label> {
        public:
            static bool const is_precomputed = true;
            precompute_kernel_problem(precompute_kernel_problem const&) = delete;
            precompute_kernel_problem & operator= (precompute_kernel_problem const&) = delete;
            precompute_kernel_problem(precompute_kernel_problem &&) = default;
            precompute_kernel_problem & operator= (precompute_kernel_problem &&) = default;

            template <class OtherProblem,
                      typename UnaryFunction,
                      typename UnaryPredicate = always>
            precompute_kernel_problem(OtherProblem && other,
                                      UnaryFunction map,
                                      UnaryPredicate filter = {})
                : basic_problem<Container, Label>(std::move(other), map, filter)
                , kernel(std::move(other.kernel))
            {
            }

            template <typename = typename std::enable_if<std::is_default_constructible<Kernel>::value>::type>
            precompute_kernel_problem (size_t dim)
                : basic_problem<Container, Label>(dim) {}

            precompute_kernel_problem (const Kernel & k, size_t dim)
                : basic_problem<Container, Label>(dim), kernel(k) {}

            template <typename ..., class L = Label,
                      typename = typename std::enable_if<traits::is_convertible_label<L>::value>::type>
            struct svm_problem generate() {
                kernel_data.clear();
                ptrs.clear();
                int i = 1;
                for (Container const& xi : orig_data) {
                    kernel_data.push_back(kernelize(xi, i));
                    ptrs.push_back(kernel_data.back().ptr());
                    ++i;
                }
                struct svm_problem p;
                p.x = ptrs.data();
                p.y = labels.data();
                p.l = labels.size();
                return p;
            }
            dataset kernelize(Container const& xi, double index = 1) const {
                std::vector<double> v;
                v.push_back(index);
                for (Container const& xj : orig_data) {
                    v.push_back(kernel(xi, xj));
                }
                return dataset(v, 0, false);
            }

            template <class OtherContainer, class OtherLabel>
            friend class basic_problem;

            template <class OtherKernel, class OtherContainer, class OtherLabel>
            friend class precompute_kernel_problem;
        private:
            using basic_problem<Container, Label>::orig_data;
            using basic_problem<Container, Label>::labels;
            Kernel kernel;
            std::vector<dataset> kernel_data;
            std::vector<struct svm_node *> ptrs;
        };

    }
}
