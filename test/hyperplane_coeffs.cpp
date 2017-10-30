#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "kernel/linear_precomputed.hpp"
#include "hyperplane_model.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>


template <class Kernel>
void hyperplane_coeffs_test (size_t N, size_t M, double eps) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

    hyperplane_model trail_model(N, rng);

    svm::problem<Kernel> prob;
    using input_t = typename svm::problem<Kernel>::input_container_type;

    int ones = 0;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        double y = trail_model(xs);
        if (y > 0)
            ++ones;
        prob.add_sample(input_t(std::move(xs)), y);
    }
    std::cout << "fraction of ones: " << 1. * ones / M << std::endl;

    svm::parameters<Kernel> params;
    svm::introspective_model<Kernel> empirical_model(std::move(prob), params);

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
        CHECK(*it == doctest::Approx(empirical_C[n]).epsilon(eps));
    }
}

TEST_CASE("hyperplane-coeffs-builtin") {
    hyperplane_coeffs_test<svm::kernel::linear>(25, 10000, 0.1);
}

TEST_CASE("hyperplane-coeffs-precomputed") {
    hyperplane_coeffs_test<svm::kernel::linear_precomputed>(25, 10000, 0.1);
}
