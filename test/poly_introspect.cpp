#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "svm-wrapper.hpp"

#include <array>
#include <iostream>


using kernel_t = svm::kernel::polynomial<2>;
using model_t = svm::model<kernel_t>;
using problem_t = svm::problem<kernel_t>;
using param_t = svm::parameters<kernel_t>;
using array_t = std::array<double, 4>;

static const array_t ys = {+1, +1, -1, -1};
static const std::array<array_t,4> xs {
    array_t {  1,  2,  3,  4 },
    array_t { -8, -6, -4, -2 },
    array_t {  4,  3,  2,  1 },
    array_t { -2, -4, -6, -8 }
};

static const model_t model = [] {
    problem_t prob(4);
    auto itX = xs.begin();
    auto itY = ys.begin();
    for (; itX != xs.end(); ++itX, ++itY)
        prob.add_sample(svm::dataset(*itX), *itY);
    return model_t(std::move(prob), param_t(1., 0.5));
} ();

static const array_t ya = [] {
    array_t ya;
    auto it = ya.begin();
    for (auto p : model) {
        std::tie(*(it), std::ignore) = p;
        std::cout << *it << std::endl;
        ++it;
    }
    return ya;
} ();

TEST_CASE("polynomial-introspect-scalar") {
    svm::tensor_introspector<kernel_t, 0> introspector(model);
    CHECK(introspector.tensor() == doctest::Approx(0.25));
}

TEST_CASE("polynomial-introspect-vector") {
    svm::tensor_introspector<kernel_t, 1> introspector(model);
    array_t u = {0, 0, 0, 0};
    auto itX = xs.begin();
    auto itYA = ya.begin();
    for (; itX != xs.end(); ++itX, ++itYA) {
        auto itU = u.begin();
        auto itXX = itX->begin();
        for (; itXX != itX->end(); ++itXX, ++itU)
            *itU += *itYA * *itXX;
    }
    for (size_t i = 0; i < 4; ++i)
        CHECK(u[i] == doctest::Approx(introspector.tensor({i})));
}

TEST_CASE("polynomial-introspect-matrix") {
    svm::tensor_introspector<kernel_t, 2> introspector(model);
    std::array<array_t, 4> u {
        array_t {0, 0, 0, 0},
        array_t {0, 0, 0, 0},
        array_t {0, 0, 0, 0},
        array_t {0, 0, 0, 0}
    };
    auto itX = xs.begin();
    auto itYA = ya.begin();
    for (; itX != xs.end(); ++itX, ++itYA) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                u[i][j] += *itYA * (*itX)[i] * (*itX)[j];
            }
        }
    }
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            CHECK(u[i][j] == doctest::Approx(introspector.tensor({i, j})));
}
