#pragma once
#include "svm-wrapper.hpp"
#include "hyperplane_model.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>

template <class Kernel>
void hyperplane_test (size_t N, size_t M, double threshold) {
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
    svm::model<Kernel> empirical_model(std::move(prob), params);

    int correct = 0;
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> xs(N);
        for (double & x : xs)
            x = uniform(rng);
        double y_true = trail_model(xs);
        double y_pred = empirical_model(input_t(std::move(xs)));
        if (y_true * y_pred > 0)
            ++correct;
    }
    double success_rate = 1. * correct / M;
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);
}
