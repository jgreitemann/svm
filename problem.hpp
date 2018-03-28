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

#include "dataset.hpp"
#include "svm.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


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

            template <class OtherProblem, typename UnaryFunction>
            basic_problem(OtherProblem && other, UnaryFunction label_map) {
                dimension = other.dimension;
                append_problem(std::move(other), label_map);
            }

            void add_sample(Container && ds, Label label) {
                orig_data.push_back(std::move(ds));
                labels.push_back(label);
            }

            void add_sample(Container const& ds, Label label) {
                orig_data.push_back(ds);
                labels.push_back(label);
            }

            template <class OtherProblem, typename UnaryFunction>
            void append_problem (OtherProblem && other, UnaryFunction label_map) {
                if (dimension != other.dimension)
                    throw std::logic_error("incompatible problem dimensions");
                orig_data.reserve(orig_data.size() + other.orig_data.size());
                orig_data.insert(orig_data.end(),
                                 std::make_move_iterator(other.orig_data.begin()),
                                 std::make_move_iterator(other.orig_data.end()));
                other.orig_data.clear();
                std::transform(other.labels.begin(),
                               other.labels.end(),
                               std::back_inserter(labels),
                               label_map);
                other.labels.clear();
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

        template <class Label>
        class patch_through_problem : public basic_problem<dataset, Label> {
        public:
            static bool const is_precomputed = false;
            patch_through_problem(size_t dim) : basic_problem<dataset, Label>(dim) {};
            patch_through_problem(patch_through_problem const&) = delete;
            patch_through_problem & operator= (patch_through_problem const&) = delete;
            patch_through_problem(patch_through_problem &&) = default;
            patch_through_problem & operator= (patch_through_problem &&) = default;

            template <class OtherProblem, typename UnaryFunction>
            patch_through_problem(OtherProblem && other, UnaryFunction map)
                : basic_problem<dataset, Label>(std::move(other), map) {}

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

        template <class Kernel, class Container, class Label>
        class precompute_kernel_problem : public basic_problem<Container, Label> {
        public:
            static bool const is_precomputed = true;
            precompute_kernel_problem(precompute_kernel_problem const&) = delete;
            precompute_kernel_problem & operator= (precompute_kernel_problem const&) = delete;
            precompute_kernel_problem(precompute_kernel_problem &&) = default;
            precompute_kernel_problem & operator= (precompute_kernel_problem &&) = default;

            template <class OtherProblem, typename UnaryFunction>
            precompute_kernel_problem(OtherProblem && other, UnaryFunction map)
                : basic_problem<dataset, Label>(std::move(other), map), kernel(std::move(other.kernel)) {}

            template <typename = typename std::enable_if<std::is_default_constructible<Kernel>::value>::type>
            precompute_kernel_problem (size_t dim)
                : basic_problem<Container, Label>(dim) {}

            precompute_kernel_problem (const Kernel & k, size_t dim)
                : basic_problem<Container, Label>(dim), kernel(k) {}

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
            dataset kernelize(Container const& xi, double index = 1) {
                std::vector<double> v;
                v.push_back(index);
                for (Container const& xj : orig_data) {
                    v.push_back(kernel(xi, xj));
                }
                return dataset(v, 0, false);
            }

            template <class OtherContainer, class OtherLabel>
            friend class basic_problem;
        private:
            using basic_problem<Container, Label>::orig_data;
            using basic_problem<Container, Label>::labels;
            Kernel kernel;
            std::vector<dataset> kernel_data;
            std::vector<struct svm_node *> ptrs;
        };

    }

    template <class Kernel, class Label = double>
    class problem : public detail::precompute_kernel_problem<Kernel, typename Kernel::input_container_type, Label> {
        using detail::precompute_kernel_problem<Kernel, typename Kernel::input_container_type, Label>::precompute_kernel_problem;
    };

}
