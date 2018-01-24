#pragma once

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace combinatorics {

    template <typename T>
    T binomial(T n, T k) {
        T c = 1, i;
        if (k > n - k)
            k = n - k;  /* take advantage of symmetry */
        for (i = 1; i <= k; i++, n--) {
            if (c / i > std::numeric_limits<T>::max() / n)
                throw std::range_error("Binomial coefficient exceeds range");
            c = c/i * n + c%i * n / i;  /* split c*n/i into (c/i*i + c%i)*n/i */
        }
        return c;
    }

    template <typename It, typename T = typename std::iterator_traits<It>::value_type>
    T multinomial_sorted(It begin, It end) {
        auto M = std::distance(begin, end);
        T n = std::accumulate(begin, end, 0);
        if (M < 2)
            throw std::runtime_error("Multinomial coefficient ill-defined for less than two k's.");
        --end;
        T c = 1, i = 1;
        auto m = M - 1;
        while (n > *end) {
            while (i > *begin) {
                --m;
                ++begin;
            }
            for (T j = 0; j < m; ++j, --n) {
                if (c / i > std::numeric_limits<T>::max() / n)
                    throw std::range_error("Multinomial coefficient exceeds range");
                c = c/i * n + c%i * n / i;
            }
            ++i;
        }
        return c;
    }

    template <typename It, typename T = typename std::iterator_traits<It>::value_type>
    T multinomial(It const& begin, It const& end) {
        if (std::is_sorted(begin, end)) {
            return multinomial_sorted(begin, end);
        } else {
            std::vector<T> sorted(begin, end);
            std::sort(sorted.begin(), sorted.end());
            return multinomial_sorted(sorted.begin(), sorted.end());
        }
    }

    template <typename Container, typename T = typename Container::value_type>
    T multinomial(Container const& ks) {
        return multinomial(std::begin(ks), std::end(ks));
    }

    template <typename T>
    T multinomial(std::initializer_list<T> il) {
        return multinomial(il.begin(), il.end());
    }

    template <typename Container, typename T = typename Container::value_type>
    T multinomial_in_place(Container & ks) {
        std::sort(std::begin(ks), std::end(ks));
        return multinomial_sorted(std::begin(ks), std::end(ks));
    }

    template <typename Base_t, typename Exp_t,
              typename = typename std::enable_if<std::is_integral<Exp_t>::value>::type>
    Base_t ipow (Base_t base, Exp_t exp) {
        if (exp < 0) {
            if (std::is_integral<Base_t>::value) {
                std::domain_error("negative exponent in exponentiation of integral base");
            } else {
                exp *= -1;
                base = 1 / base;
            }
        }
        Base_t result = 1;
        while (exp) {
            if (exp & 1) {
                result *= base;
                if (result > std::numeric_limits<Base_t>::max() / base)
                    throw std::range_error("overflow in ipow");
            }
            exp >>= 1;
            if (base > std::numeric_limits<Base_t>::max() / base)
                throw std::range_error("overflow in ipow");
            base *= base;
        }
        return result;
    }

}
