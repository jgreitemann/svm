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

#include <limits>
#include <stdexcept>


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

}
