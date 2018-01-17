#pragma once

#include "dataset.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>


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
