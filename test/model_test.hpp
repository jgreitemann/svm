#pragma once
#include "svm-wrapper.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>

template <class Problem, class TrialModel, class RNG_t>
Problem fill_problem (size_t M, RNG_t & rng,
                      TrialModel const& trial_model)
{
    std::uniform_real_distribution<double> uniform;
    Problem prob(trial_model.dim());
    using input_t = typename Problem::input_container_type;

    int ones = 0;
    double y;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(trial_model.dim());
        for (double & x : xs)
            x = uniform(rng);
        std::tie(y, std::ignore) = trial_model(xs);
        if (y > 0)
            ++ones;
        prob.add_sample(input_t(std::move(xs)), y);
    }
    std::cout << "fraction of ones: " << 1. * ones / M << std::endl;
    return prob;
}

template <class ModelA, class ModelB, class RNG>
double test_model (size_t M, RNG & rng,
                   ModelA & model_a,
                   ModelB & model_b)
{
    std::uniform_real_distribution<double> uniform;
    using input_a_t = typename ModelA::input_container_type;
    using input_b_t = typename ModelB::input_container_type;

    CHECK(model_a.dim() == model_b.dim());
    size_t N = model_a.dim();

    int correct = 0;
    double y_a, y_b;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        std::tie(y_a, std::ignore) = model_a(input_a_t(xs));
        std::tie(y_b, std::ignore) = model_b(input_b_t(std::move(xs)));
        if (y_a * y_b > 0)
            ++correct;
    }
    return 1. * correct / M;
}

template <class Kernel, class TrialModel, class RNG_t = std::mt19937>
void model_test (size_t M, double threshold, TrialModel const& trial_model, RNG_t rng = RNG_t(42)) {
    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(M, rng, trial_model),
        params);

    double success_rate = test_model(M, rng, trial_model, empirical_model);
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);
}
