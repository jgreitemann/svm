#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "kernel/linear_precomputed.hpp"

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

template <class Kernel>
void hyperplane_test (size_t N, size_t M, double threshold) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

    hyperplane_model trail_model(N, rng);

    svm::problem<Kernel> prob;

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

    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(std::move(prob), params);

    int correct = 0;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        double y_true = trail_model(xs);
        double y_pred = empirical_model(svm::dataset(xs));
        if (y_true * y_pred > 0)
            ++correct;
    }
    double success_rate = 1. * correct / M;
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);
}

TEST_CASE("hyperplane-builtin") {
    hyperplane_test<svm::kernel::linear>(25, 2500, 0.98);
}

TEST_CASE("hyperplane-precomputed") {
    hyperplane_test<svm::kernel::linear_precomputed>(25, 2500, 0.98);
}

TEST_CASE("hyperplane-coeffs") {
    size_t M = 10000;
    size_t N = 7;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

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
    svm::model<kernel_t> empirical_model(std::move(prob), params);

    auto it = trail_model.coefficients().begin();
    for (size_t n = 0; n < N; ++n, ++it) {
        CHECK(*it == doctest::Approx(empirical_model.C(n)).epsilon(0.1));
    }
}
