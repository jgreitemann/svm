#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "model_test.hpp"
#include "hyperplane_model.hpp"

#include <random>


TEST_CASE("hyperplane-sigmoid") {
    std::mt19937 rng(42);
    hyperplane_model trial_model(25, rng);
    model_test<svm::kernel::sigmoid>(2500, 0.94, trial_model, rng);
}
