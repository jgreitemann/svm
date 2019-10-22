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

#include <cstddef>
#include <limits>
#include <type_traits>


namespace svm {

    static constexpr size_t DYNAMIC = std::numeric_limits<size_t>::max();

    namespace traits {

        template <typename...>
        using void_t = void;

        template <class Label>
        struct is_convertible_label
            : std::integral_constant<bool,
                                     bool(std::is_convertible<Label, double>::value)
                                     && bool(std::is_convertible<double, Label>::value)
                                     > {};

        template <class Label, class = void_t<>>
        struct label_size : std::integral_constant<size_t, DYNAMIC> {};

        template <class Label>
        struct label_size<Label, void_t<decltype(Label::nr_labels)>>
            : std::integral_constant<size_t, Label::nr_labels> {};

        template <class Label>
        struct label_traits {
            static const size_t label_dim = Label::label_dim;
            static const size_t nr_labels = label_size<Label>::value;
            static auto begin (Label const& l) {
                return l.begin();
            }
            static auto end (Label const& l) {
                return l.end();
            }
            template <class Iterator>
            static Label from_iterator (Iterator begin) {
                return Label(begin);
            }
        };

        template <>
        struct label_traits<double> {
            static const size_t label_dim = 1;
            static const size_t nr_labels = DYNAMIC;
            static double const * begin (double const& l) { return &l; }
            static double const* end (double const& l) { return &l + 1; }
            template <class Iterator>
            static double from_iterator (Iterator begin) {
                return *begin;
            }
        };

        template <typename Label>
        using is_dynamic_label = std::integral_constant<bool, (label_traits<Label>::nr_labels == DYNAMIC)>;

        template <typename Label>
        using is_binary_label = std::integral_constant<bool, (label_traits<Label>::nr_labels == 2)>;

    }
}
