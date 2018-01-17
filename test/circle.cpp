#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "model_test.hpp"
#include "circle_model.hpp"

#include <random>


TEST_CASE("circle-poly2-center") {
    std::mt19937 rng(42);
    circle_model trial_model({0., 0.}, 0.7, 2);
    model_test<svm::kernel::polynomial<2>>(1000, 0.98, trial_model, rng, 0.01);
}

TEST_CASE("circle-poly2-offset") {
    std::mt19937 rng(42);
    circle_model trial_model({0.3, 0.2}, 0.3);
    model_test<svm::kernel::polynomial<2>>(1000, 0.9, trial_model, rng, 0.45);
}

TEST_CASE("circle-rbf") {
    std::mt19937 rng(42);
    circle_model trial_model({0.3, 0.2}, 0.3);
    model_test<svm::kernel::rbf>(1000, 0.98, trial_model, rng, 0.01);
}
