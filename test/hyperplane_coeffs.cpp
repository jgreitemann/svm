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

    hyperplane_model trial_model(N, rng);

    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(N, M, rng, trial_model),
        params);
    using input_t = typename svm::model<Kernel>::input_container_type;
    svm::linear_introspector<Kernel> introspector(empirical_model);

    std::vector<double> empirical_C(N);
    for (size_t i = 0; i < N; ++i)
        empirical_C[i] = introspector.coefficient(i);
    double norm_trial = 0, norm_emp = 0;
    for (double c : trial_model.coefficients())
        norm_trial += c * c;
    for (double c : empirical_C)
        norm_emp += c * c;

    // compare decision function values from model with those
    // manually calculated from the inferred hyperplane coeffs
    std::uniform_real_distribution<double> uniform;
    double d_pred, d_calc;
    for (size_t m = 0; m < 25; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        d_calc = 0;
        auto itC = empirical_C.begin();
        for (double x : xs) {
            d_calc += *itC * x;
            ++itC;
        }
        d_calc -= empirical_model.rho();
        std::tie(std::ignore, d_pred) = empirical_model(input_t(std::move(xs)));
        CHECK(d_calc == doctest::Approx(d_pred));
    }

    // compare empirical coeffs with input trial coeffs
    for (double & c : empirical_C)
        c *= sqrt(norm_trial / norm_emp);
    auto it = trial_model.coefficients().begin();
    for (size_t n = 0; n < N; ++n, ++it) {
        CHECK(*it == doctest::Approx(empirical_C[n]).epsilon(eps));
    }
}

TEST_CASE("hyperplane-coeffs-builtin") {
    hyperplane_coeffs_test<svm::kernel::linear>(25, 10000, 0.1);
}

TEST_CASE("hyperplane-coeffs-precomputed") {
    hyperplane_coeffs_test<svm::kernel::linear_precomputed>(25, 10000, 0.1);
}
