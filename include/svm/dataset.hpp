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

#include <initializer_list>
#include <iterator>
#include <vector>

#include <svm/libsvm/svm.h>


namespace svm {

    // forward declaration
    class dataset;

    class data_view {
    public:
        class const_iterator {
        public:
            typedef std::ptrdiff_t difference_type;
            typedef double value_type;
            typedef double * pointer;
            typedef double & reference;
            typedef std::input_iterator_tag iterator_category;

            const_iterator (struct svm_node const * ptr, int index = 1)
                : ptr(ptr), index(index) {}

            const_iterator & operator++ () {
                if (ptr->index == index)
                    ++ptr;
                ++index;
                return *this;
            }

            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }

            double operator* () const {
                if (ptr->index == index)
                    return ptr->value;
                return 0.;
            }

            bool is_end () const {
                return index == -1 || ptr->index == -1;
            }

            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                if (lhs.is_end() && rhs.is_end())
                    return true;
                return lhs.index == rhs.index && lhs.ptr == rhs.ptr;
            }

            friend bool operator!= (const_iterator lhs, const_iterator rhs) {
                return !(lhs == rhs);
            }

        private:
            struct svm_node const * ptr;
            int index;
        };

        data_view ()
            : begin_ptr(nullptr), start_index(-1) {}

        data_view (struct svm_node const * ptr, int start_index = 1)
            : begin_ptr(ptr), start_index(start_index) {}

        data_view (dataset const& ds); // forward declaration

        const_iterator begin () const {
            if (start_index == -1)
                return end();
            return const_iterator(begin_ptr, start_index);
        }

        const_iterator end () const {
            return const_iterator(nullptr, -1);
        }

        double front () const {
            return *begin();
        }

        double dot (data_view other) const {
            data_view::const_iterator lhs = begin();
            data_view::const_iterator rhs = other.begin();
            double sum = 0;
            for (; lhs != end() && rhs != other.end(); ++lhs, ++rhs) {
                sum += *lhs * *rhs;
            }
            return sum;
        }

    private:
        struct svm_node const * begin_ptr;
        int start_index;
    };

    inline double dot (data_view lhs, data_view rhs) {
        return lhs.dot(rhs);
    }


    class dataset {
    public:
        using const_iterator = data_view::const_iterator;

        template <typename OutputIterator>
        dataset (OutputIterator begin, OutputIterator end,
                 int start_index = 1, bool skip_zeros = true)
            : start_index(start_index)
        {
            nodify(begin, end, skip_zeros);
        }

        template <typename Container>
        dataset (Container const& c,
                 int start_index = 1, bool skip_zeros = true)
            : start_index(start_index)
        {
            nodify(c.begin(), c.end(), skip_zeros);
        }

        dataset (std::initializer_list<double> il)
            : start_index(1)
        {
            nodify(il.begin(), il.end(), true);
        }

        struct svm_node * ptr () {
            return data_.data();
        }

        struct svm_node const * ptr () const {
            return data_.data();
        }

        std::vector<struct svm_node> const& data () const {
            return data_;
        }

        data_view view () const {
            return data_view(ptr(), start_index);
        }

        const_iterator begin () const {
            return view().begin();
        }

        const_iterator end () const {
            return view().end();
        }

    private:
        template <typename OutputIterator>
        void nodify (OutputIterator begin, OutputIterator end, bool skip_zeros) {
            for (int i = start_index; begin != end; ++i, ++begin)
                if (!skip_zeros || *begin != 0)
                    data_.push_back({ .index = i, .value = static_cast<double>(*begin) });
            data_.push_back({ .index = -1, .value = {} });
        }

        std::vector<struct svm_node> data_;
        int start_index;
    };

    inline data_view::data_view (dataset const& ds) {
        *this = ds.view();
    }

}
