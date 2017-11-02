#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "kernel/linear_precomputed.hpp"
#include "hyperplane_model.hpp"
#include "hyperplane_test.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>


template <class Kernel>
void hyperplane_coeffs_test (size_t N, size_t M, double eps) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform;

    hyperplane_model trial_model(N, rng);

    svm::parameters<Kernel> params;
    svm::introspective_model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(N, M, rng, trial_model),
        params);

    std::vector<double> empirical_C = empirical_model.coefficients();
    double norm_trial = 0, norm_emp = 0;
    for (double c : trial_model.coefficients())
        norm_trial += c * c;
    for (double c : empirical_C)
        norm_emp += c * c;
    for (double & c : empirical_C)
        c *= sqrt(norm_trial / norm_emp);
    auto it = trial_model.coefficients().begin();
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
