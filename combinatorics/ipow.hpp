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
#include <type_traits>


namespace combinatorics {

    template <typename Base_t, typename Exp_t,
              typename = typename std::enable_if<std::is_integral<Exp_t>::value>::type>
    Base_t ipow (Base_t base, Exp_t exp) {
        if (exp < 0) {
            if (std::is_integral<Base_t>::value) {
                throw std::domain_error("negative exponent in exponentiation of integral base");
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
