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

#include <array>
#include <cstdlib>
#include <iterator>
#include <utility>


namespace svm {

    namespace detail {

        template <typename ContiguousContainer>
        struct container_factory {
            using value_type = typename ContiguousContainer::value_type;
            using size_type = typename ContiguousContainer::size_type;

            static ContiguousContainer create(size_t size) {
                return ContiguousContainer(size);
            }

            template <typename ContiguousIterator>
            static ContiguousContainer copy(ContiguousIterator begin,
                                            ContiguousIterator end)
            {
                return ContiguousContainer(begin, end);
            }

            template <typename OtherContainer>
            static ContiguousContainer copy(OtherContainer const& other) {
                return ContiguousContainer(
                    container_factory<OtherContainer>::ptr(other),
                    container_factory<OtherContainer>::ptr(other)
                    + container_factory<OtherContainer>::size(other));
            }

            static value_type* ptr(ContiguousContainer& c) {
                return c.data();
            }

            static value_type const* ptr(ContiguousContainer const& c) {
                return c.data();
            }

            static size_type size(ContiguousContainer const& c) {
                return c.size();
            }
        };

        template <typename T, size_t N>
        struct container_factory<std::array<T, N>> {
            using value_type = typename std::array<T, N>::value_type;
            using size_type = typename std::array<T, N>::size_type;

            static std::array<T,N> create(size_t) {
                return {};
            }

            template <typename ContiguousIterator>
            static std::array<T,N> copy(ContiguousIterator begin,
                                        ContiguousIterator end)
            {
                using Indices = std::make_index_sequence<N>;
                if (std::distance(begin, end) != N)
                    throw std::invalid_argument("specified range does not "
                                                "match array size");
                return copy_impl(begin, Indices{});
            }

            template <typename OtherContainer>
            static std::array<T, N> copy(OtherContainer const& other) {
                using Indices = std::make_index_sequence<N>;
                return copy_impl(container_factory<OtherContainer>::ptr(other),
                                 Indices{});
            }

            static T* ptr(std::array<T,N>& c) {
                return c.data();
            }

            static T const* ptr(std::array<T,N> const& c) {
                return c.data();
            }

            static size_type size(std::array<T,N> const&) {
                return N;
            }
        private:
            template <typename ContiguousIterator,
                      size_t... I>
            static std::array<T,N> copy_impl(ContiguousIterator begin,
                                             std::index_sequence<I...>) {
                return {begin[I]...};
            }
        };

        template <>
        struct container_factory<double> {
            using value_type = double;
            using size_type = size_t;

            static double create(size_t) {
                return {};
            }

            template <typename ContiguousIterator>
            static double copy(ContiguousIterator begin,
                               ContiguousIterator end)
            {
                if (std::distance(begin, end) != 1)
                    throw std::invalid_argument("specified range exceeds "
                                                "one element");
                return *begin;
            }

            template <typename OtherContainer>
            static double copy(OtherContainer const& other) {
                if (container_factory<OtherContainer>::size(other) != 1)
                    throw std::invalid_argument("specified container has more "
                                                "than one element");
                return *container_factory<OtherContainer>::ptr(other);
            }

            static double* ptr(double& c) {
                return &c;
            }

            static double const* ptr(double const& c) {
                return &c;
            }

            static size_type size(double const&) {
                return 1;
            }
        };

    }

}
