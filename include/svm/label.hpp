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

#include <iostream>
#include <iterator>
#include <stdexcept>


#define SVM_LABEL_BEGIN(LABELNAME, LABELCOUNT)                          \
namespace LABELNAME {                                                   \
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
        unsigned short val;                                             \
    };                                                                  \
    static short i = 0;

#define SVM_LABEL_ADD(OPTIONNAME)                       \
    static const label OPTIONNAME { i++, #OPTIONNAME };

#define SVM_LABEL_END() }
