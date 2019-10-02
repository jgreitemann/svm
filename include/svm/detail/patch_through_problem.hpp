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

        template <class Label>
        class patch_through_problem : public basic_problem<dataset, Label> {
        public:
            static bool const is_precomputed = false;
            patch_through_problem(size_t dim) : basic_problem<dataset, Label>(dim) {};
            patch_through_problem(patch_through_problem const&) = delete;
            patch_through_problem & operator= (patch_through_problem const&) = delete;
            patch_through_problem(patch_through_problem &&) = default;
            patch_through_problem & operator= (patch_through_problem &&) = default;

            template <class OtherProblem,
                      typename UnaryFunction,
                      typename UnaryPredicate = always>
            patch_through_problem(OtherProblem && other,
                                  UnaryFunction map,
                                  UnaryPredicate filter = {})
                : basic_problem<dataset, Label>(std::move(other), map, filter)
            {
            }

            template <typename ..., typename L = Label,
                      typename = typename std::enable_if<traits::is_convertible_label<L>::value>::type>
            struct svm_problem generate() {
                ptrs.clear();
                for (dataset & ds : orig_data)
                    ptrs.push_back(ds.ptr());
                raw_labels.clear();
                for (Label const& l : labels)
                    raw_labels.push_back(l);
                struct svm_problem p;
                p.x = ptrs.data();
                p.y = raw_labels.data();
                p.l = raw_labels.size();
                return p;
            }

            template <class OtherContainer, class OtherLabel>
            friend class basic_problem;
        private:
            using basic_problem<dataset, Label>::orig_data;
            using basic_problem<dataset, Label>::labels;
            std::vector<struct svm_node *> ptrs;
            std::vector<double> raw_labels;
        };

    }
}
