#pragma once

#include <limits>

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
