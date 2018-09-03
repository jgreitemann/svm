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

#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>


namespace svm {

    static const size_t DYNAMIC = std::numeric_limits<size_t>::max();

    namespace traits {
        template <class Label>
        struct is_convertible_label
            : std::integral_constant<bool,
                                     bool(std::is_convertible<Label, double>::value)
                                     && bool(std::is_convertible<double, Label>::value)
                                     > {};

        template <class Label>
        struct label_traits {
            static const size_t label_dim = Label::label_dim;
            static const size_t nr_labels = Label::nr_labels;
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


#define SVM_LABEL_BEGIN(LABELNAME, LABELCOUNT)                          \
    namespace LABELNAME {                                               \
    static const char * NAMES[(LABELCOUNT)];                            \
    static double FLOAT_REPRS[(LABELCOUNT)];                            \
    struct label {                                                      \
        static const size_t nr_labels = (LABELCOUNT);                   \
        static const size_t label_dim = 1;                              \
        label () : val(FLOAT_REPRS[0] + 0.5) {}                         \
        label (int val) : val(val) {}                                   \
        label (int val, const char * c_str) : val(val) {                \
            NAMES[val] = c_str;                                         \
            FLOAT_REPRS[val] = double(val);                             \
        }                                                               \
        template <class Iterator,                                       \
                  typename Tag = typename std::iterator_traits<Iterator>::value_type> \
        label (Iterator begin) : val (*begin + 0.5) {                   \
            if (val < 0 || val >= nr_labels)                            \
                throw std::runtime_error("invalid label");              \
        }                                                               \
        label (double x) : val (x + 0.5) {                              \
            if (x < 0 || x >= nr_labels)                                \
                throw std::runtime_error("invalid label");              \
        }                                                               \
        operator double() const { return val; }                         \
        double const * begin() const { return &FLOAT_REPRS[val]; }      \
        double const * end() const { return &FLOAT_REPRS[val] + 1; }    \
        friend bool operator== (label lhs, label rhs) {                 \
            return lhs.val == rhs.val;                                  \
        }                                                               \
        friend std::ostream & operator<< (std::ostream & os, label l) { \
            return os << NAMES[l.val];                                  \
        }                                                               \
    private:                                                            \
        short val;                                                      \
    };                                                                  \
    static short i = 0;

#define SVM_LABEL_ADD(OPTIONNAME)                       \
    static const label OPTIONNAME { i++, #OPTIONNAME };

#define SVM_LABEL_END() }
