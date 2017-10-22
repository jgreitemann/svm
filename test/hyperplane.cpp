#include "svm-wrapper.hpp"

#include <iostream>
#include <random>
#include <stdexcept>


class hyperplane_model {
public:
    template <class RNG>
    hyperplane_model (size_t N, RNG & rng) : coeffs(N) {
        std::uniform_int_distribution<int> dice(-6, 6);
        for (double & c : coeffs)
            c = dice(rng);
    }

    double operator() (std::vector<double> const& xs) const {
        double sum = 0;
        auto it_x = xs.begin();
        auto it_c = coeffs.begin();
        for (; it_x != xs.end() && it_c != coeffs.end(); ++it_x, ++it_c)
            sum += *it_x * *it_c;
        if (it_x != xs.end() || it_c != coeffs.end())
            throw std::length_error("dimensions don't match");
        return sum > 0 ? 1. : -1.;
    }

    std::vector<double> const& coefficients () const {
        return coeffs;
    }
private:
    std::vector<double> coeffs;
};

int main () {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

    size_t N = 25;
    size_t M = 10000;

    hyperplane_model trail_model(N, rng);

    typedef svm::kernel::linear kernel_t;
    svm::problem<kernel_t> prob;

    int ones = 0;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        double y = trail_model(xs);
        if (y > 0)
            ++ones;
        prob.add_sample(svm::dataset(xs), y);
    }
    std::cout << "fraction of ones: " << 1. * ones / M << std::endl;

    svm::parameters<kernel_t> params;
    svm::model<kernel_t> model(std::move(prob), params);
}
