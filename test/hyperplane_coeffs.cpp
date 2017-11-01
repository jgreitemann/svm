#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "hyperplane_model.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>


TEST_CASE("hyperplane-coeffs") {
    size_t M = 10000;
    size_t N = 25;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

    hyperplane_model trail_model(N, rng);

    typedef svm::kernel::linear kernel_t;
    svm::problem<kernel_t> prob(N);

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
    svm::introspective_model<kernel_t> empirical_model(std::move(prob), params);

    std::vector<double> empirical_C = empirical_model.coefficients();
    double norm_trail = 0, norm_emp = 0;
    for (double c : trail_model.coefficients())
        norm_trail += c * c;
    for (double c : empirical_C)
        norm_emp += c * c;
    for (double & c : empirical_C)
        c *= sqrt(norm_trail / norm_emp);
    auto it = trail_model.coefficients().begin();
    for (size_t n = 0; n < N; ++n, ++it) {
        std::cout << *it << '\t' << empirical_C[n] << std::endl;
        CHECK(*it == doctest::Approx(empirical_C[n]).epsilon(0.1));
    }
}
