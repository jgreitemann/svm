#pragma once

#include <combinatorics/multinomial.hpp>

#include <algorithm>
#include <iterator>
#include <map>
#include <utility>


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
