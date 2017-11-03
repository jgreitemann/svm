#pragma once

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>


class hyperplane_model {
public:
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
private:
    std::vector<double> coeffs;
};
