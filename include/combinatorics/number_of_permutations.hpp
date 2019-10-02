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
#include <map>
#include <utility>

#include <combinatorics/multinomial.hpp>


namespace combinatorics {

    template <typename It>
    size_t number_of_permutations(It begin, It end) {
        using T = typename std::iterator_traits<It>::value_type;
        std::map<T, size_t> buckets;
        for (; begin != end; ++begin) {
            auto it = buckets.find(*begin);
            if (it == buckets.end()) {
                buckets[*begin] = 1;
            } else {
                ++(it->second);
            }
        }
        std::vector<size_t> counts;
        std::transform(buckets.begin(), buckets.end(), std::back_inserter(counts),
                       [] (std::pair<T const&,size_t const&> p) { return p.second; });
        return multinomial(counts);
    }

    template <typename Container>
    size_t number_of_permutations(Container const& c) {
        return number_of_permutations(std::begin(c), std::end(c));
    }

}
