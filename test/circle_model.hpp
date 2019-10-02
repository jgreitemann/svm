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

#include <cmath>
#include <iterator>
#include <utility>

#include <svm/dataset.hpp>


class circle_model {
public:
    typedef svm::dataset input_container_type;

    circle_model (svm::dataset const& c, double r, size_t dim = 0)
        : c(c), gamma(log(2.)/r/r), c2(dot(c, c)), dim_(dim) {
        if (dim_ == 0)
            dim_ = std::distance(c.begin(), c.end());
    }

    std::pair<double, double> operator() (input_container_type const& x) const {
        double res = 2 * exp(-gamma * (dot(x, x) + c2 - 2. * dot(x, c))) - 1;
        return std::make_pair(res > 0 ? 1. : -1., res);
    }

    size_t dim () const {
        return dim_;
    }

private:
    svm::dataset c;
    double gamma, c2;
    size_t dim_;
};
