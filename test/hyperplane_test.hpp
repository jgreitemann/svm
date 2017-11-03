#pragma once
#include "svm-wrapper.hpp"
#include "hyperplane_model.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>

template <class Problem, class RNG>
Problem fill_problem (size_t N, size_t M, RNG & rng,
                      hyperplane_model const& trial_model)
{
    std::uniform_real_distribution<double> uniform;
    Problem prob(N);
    using input_t = typename Problem::input_container_type;

    int ones = 0;
    double y;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
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

template <class Model, class RNG>
double test_model (size_t N, size_t M, RNG & rng,
                   hyperplane_model const& trial_model,
                   Model & empirical_model)
{
    std::uniform_real_distribution<double> uniform;
    using input_t = typename Model::input_container_type;

    int correct = 0;
    double y_pred, y_true;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        std::tie(y_true, std::ignore) = trial_model(xs);
        std::tie(y_pred, std::ignore) = empirical_model(input_t(std::move(xs)));
        if (y_true * y_pred > 0)
            ++correct;
    }
    return 1. * correct / M;
}

template <class Kernel>
void hyperplane_test (size_t N, size_t M, double threshold) {
    std::mt19937 rng(42);

    hyperplane_model trial_model(N, rng);
    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(N, M, rng, trial_model),
        params);

    double success_rate = test_model(N, M, rng, trial_model, empirical_model);
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);
}
