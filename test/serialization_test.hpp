#pragma once

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "hyperplane_model.hpp"
#include "model_test.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>


template <class Kernel, class Tag>
void serializer_test (size_t N, size_t M, double threshold, std::string const& name) {
    std::mt19937 rng(42);

    hyperplane_model trial_model(N, rng);
    svm::parameters<Kernel> params;
    svm::model<Kernel> empirical_model(
        fill_problem<svm::problem<Kernel>>(M, rng, trial_model),
        params);

    svm::model_serializer<Tag, svm::model<Kernel>> saver(empirical_model);
    saver.save(name);

    svm::model<Kernel> restored_model;
    svm::model_serializer<Tag, svm::model<Kernel>> loader(restored_model);
    loader.load(name);

    double success_rate = test_model(M, rng, trial_model, restored_model);
    std::cout << "success rate: " << 100. * success_rate << "%\n";
    CHECK(success_rate > threshold);

    success_rate = test_model(M, rng, empirical_model, restored_model);
    CHECK(success_rate == doctest::Approx(1.));
}
