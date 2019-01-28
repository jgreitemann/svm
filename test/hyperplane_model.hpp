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

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>


class hyperplane_model {
public:
    typedef std::vector<double> input_container_type;

    template <class RNG>
    hyperplane_model (size_t N, RNG & rng) : coeffs(N) {
        std::uniform_int_distribution<int> dice(-6, 6);
        for (double & c : coeffs)
            c = dice(rng);
    }

    std::pair<double, double> operator() (std::vector<double> const& xs) const {
        double sum = 0;
        auto it_x = xs.begin();
        auto it_c = coeffs.begin();
        for (; it_x != xs.end() && it_c != coeffs.end(); ++it_x, ++it_c)
            sum += *it_x * *it_c;
        if (it_x != xs.end() || it_c != coeffs.end())
            throw std::length_error("dimensions don't match");
        return std::make_pair(sum > 0 ? 1. : -1., sum);
    }

    std::vector<double> const& coefficients () const {
        return coeffs;
    }

    size_t dim () const {
        return coeffs.size();
    }
private:
    std::vector<double> coeffs;
};
