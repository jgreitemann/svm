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
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include <svm/detail/always.hpp>
#include <svm/libsvm/svm.h>


namespace svm {
    namespace detail {

        template <class Container, class Label>
        class basic_problem {
        public:
            typedef Container input_container_type;
            typedef Label label_type;

            basic_problem(size_t dim) : dimension(dim) {};
            basic_problem(basic_problem const&) = delete;
            basic_problem & operator= (basic_problem const&) = delete;
            basic_problem(basic_problem &&) = default;
            basic_problem & operator= (basic_problem &&) = default;

            template <class OtherProblem,
                      typename UnaryFunction,
                      typename UnaryPredicate = always>
            basic_problem(OtherProblem && other,
                          UnaryFunction label_map,
                          UnaryPredicate filter = {})
            {
                dimension = other.dimension;
                append_problem(std::move(other), label_map, filter);
            }

            void add_sample(Container && ds, Label label) {
                orig_data.push_back(std::move(ds));
                labels.push_back(label);
            }

            void add_sample(Container const& ds, Label label) {
                orig_data.push_back(ds);
                labels.push_back(label);
            }

            template <class OtherProblem,
                      typename UnaryFunction,
                      typename UnaryPredicate = always>
            void append_problem (OtherProblem && other,
                                 UnaryFunction label_map,
                                 UnaryPredicate filter = {})
            {
                if (dimension != other.dimension)
                    throw std::logic_error("incompatible problem dimensions");

                // conditionally copy transformed labels
                std::vector<Label> transformed_labels;
                transformed_labels.reserve(other.labels.size());
                std::transform(other.labels.begin(),
                    other.labels.end(),
                    std::back_inserter(transformed_labels),
                    label_map);
                std::copy_if(transformed_labels.begin(),
                    transformed_labels.end(),
                    std::back_inserter(labels),
                    filter);
                other.labels.clear();

                // conditionally copy data accordingly
                orig_data.reserve(labels.size());
                auto label_it = transformed_labels.begin();
                std::copy_if(std::make_move_iterator(other.orig_data.begin()),
                    std::make_move_iterator(other.orig_data.end()),
                    std::back_inserter(orig_data),
                    [&](Container const&) {
                        return filter(*(label_it++));
                    });
                other.orig_data.clear();
            }

            void append_problem (basic_problem && other) {
                append_problem(std::move(other),
                               [] (Label x) -> Label {
                                   return x;
                               });
            }

            std::pair<Container const&, Label> operator[] (size_t i) const {
                return std::pair<Container const&, Label>(orig_data[i], labels[i]);
            }

            size_t size () const {
                return orig_data.size();
            }

            size_t dim () const {
                return dimension;
            }

            template <typename UnaryFunction>
            void map_labels (UnaryFunction const& map) {
                std::transform(labels.begin(), labels.end(), labels.begin(), map);
            }

            template <class OtherContainer, class OtherLabel>
            friend class basic_problem;
        protected:
            std::vector<Container> orig_data;
            std::vector<Label> labels;
        private:
            size_t dimension;
        };

    }
}
