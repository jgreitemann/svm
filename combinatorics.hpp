#pragma once

#include <limits>
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
