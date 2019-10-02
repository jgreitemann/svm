/*   Support Vector Machine Library Wrappers
 *   Copyright (C) 2018-2019  Jonas Greitemann
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program, see the file entitled "LICENCE" in the
 *   repository's root directory, or see <http://www.gnu.org/licenses/>.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <svm/label.hpp>
#include <svm/model.hpp>
#include <svm/parameters.hpp>
#include <svm/problem.hpp>
#include <svm/detail/basic_problem.hpp>
#include <svm/kernel/linear.hpp>
#include <svm/traits/label_traits.hpp>


using svm::detail::basic_problem;

SVM_LABEL_BEGIN(directions, 2)
SVM_LABEL_ADD(LEFT)
SVM_LABEL_ADD(RIGHT)
SVM_LABEL_END()

SVM_LABEL_BEGIN(quadrants, 4)
SVM_LABEL_ADD(NORTH_EAST)
SVM_LABEL_ADD(NORTH_WEST)
SVM_LABEL_ADD(SOUTH_WEST)
SVM_LABEL_ADD(SOUTH_EAST)
SVM_LABEL_END()

static_assert(!svm::traits::is_dynamic_label<quadrants::label>::value);

TEST_CASE("static-binary-class") {
    using label_t = directions::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, label_t>;
    using C = typename problem_t::input_container_type;

    const size_t M = 10000;
    const size_t N = 2;
    const size_t NC = 1;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob1(2);
    problem_t prob2(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        auto l = [c] {
            double angle = std::arg(c);
            if (angle < 0)
                return std::make_pair(directions::LEFT, directions::RIGHT);
            return std::make_pair(directions::RIGHT, directions::LEFT);
        } ();
        prob1.add_sample(C {c.real(), c.imag()}, l.first);
        prob2.add_sample(C {c.real(), c.imag()}, l.second);
    }

    using model_t = svm::model<kernel_t, label_t>;
    model_t model1(std::move(prob1), svm::parameters<kernel_t> {0.01});
    model_t model2(std::move(prob2), svm::parameters<kernel_t> {0.01});
    CHECK(model1.nr_labels() == N);
    CHECK(model2.nr_classifiers() == NC);

    std::vector<label_t> all_labels = {directions::LEFT, directions::RIGHT};
    std::sort(all_labels.begin(), all_labels.end());

    auto labels = model1.labels();
    static_assert(std::is_same<decltype(labels), std::array<label_t, N>>::value);
    CHECK(std::equal(all_labels.begin(), all_labels.end(), labels.begin()));
    labels = model2.labels();
    CHECK(std::equal(all_labels.begin(), all_labels.end(), labels.begin()));

    auto classifier1 = model1.classifier();
    auto classifier2 = model2.classifier();
    CHECK(classifier1.labels().first == labels[0]);
    CHECK(classifier1.labels().second == labels[1]);
    CHECK(classifier2.labels().first == labels[0]);
    CHECK(classifier2.labels().second == labels[1]);
    {
        auto clss1 = model1.classifier(labels[0], labels[1]);
        CHECK(clss1.labels().first == labels[0]);
        CHECK(clss1.labels().second == labels[1]);
        auto clss2 = model2.classifier(labels[0], labels[1]);
        CHECK(clss2.labels().first == labels[0]);
        CHECK(clss2.labels().second == labels[1]);
        auto clssr1 = model1.classifier(labels[1], labels[0]);
        CHECK(clssr1.labels().first == labels[1]);
        CHECK(clssr1.labels().second == labels[0]);
        auto clssr2 = model2.classifier(labels[1], labels[0]);
        CHECK(clssr2.labels().first == labels[1]);
        CHECK(clssr2.labels().second == labels[0]);
    }

    for (size_t i = 0; i < 25; ++i) {
        C c {uniform(rng), uniform(rng)};
        auto res1 = model1(c);
        auto res2 = model2(c);
        static_assert(std::is_same<decltype(res1.second),
                                   double>::value);
        auto cres1 = classifier1(c);
        auto cres2 = classifier2(c);
        CHECK(res1.first == cres1.first);
        CHECK(res1.second == cres1.second);
        CHECK(res2.first == cres2.first);
        CHECK(res2.second == cres2.second);
        CHECK(res1.first != res2.first);
        CHECK(res1.second == -res2.second);
        auto cresl1 = model1.classifier(classifier1.labels().first,
                                        classifier1.labels().second)(c);
        auto cresl2 = model2.classifier(classifier2.labels().first,
                                        classifier2.labels().second)(c);
        CHECK(res1.first == cresl1.first);
        CHECK(res1.second == cresl1.second);
        CHECK(res2.first == cresl2.first);
        CHECK(res2.second == cresl2.second);
        CHECK(cresl1.first != cresl2.first);
        CHECK(cresl1.second == -cresl2.second);
        auto cresr1 = model1.classifier(classifier1.labels().second,
                                        classifier1.labels().first)(c);
        auto cresr2 = model2.classifier(classifier2.labels().second,
                                        classifier2.labels().first)(c);
        CHECK(res1.first == cresr1.first);
        CHECK(res1.second == -cresr1.second);
        CHECK(res2.first == cresr2.first);
        CHECK(res2.second == -cresr2.second);
        CHECK(cresr1.first != cresr2.first);
        CHECK(res1.second == cresr2.second);
    }

    auto rho1 = model1.rho();
    auto rho2 = model2.rho();
    static_assert(std::is_same<decltype(rho1), double>::value);
    CHECK(rho1 == model1.classifier().rho());
    CHECK(rho2 == model2.classifier().rho());
    CHECK(rho1 == -rho2);
}

TEST_CASE("static-multi-class") {
    using label_t = quadrants::label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, label_t>;
    using C = typename problem_t::input_container_type;

    const size_t M = 10000;
    const size_t N = 4;
    const size_t NC = 6;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        label_t l = [c] {
            double angle = std::arg(c);
            if (angle < -M_PI/2) return quadrants::SOUTH_WEST;
            if (angle < 0) return quadrants::SOUTH_EAST;
            if (angle <  M_PI/2) return quadrants::NORTH_EAST;
            return quadrants::NORTH_WEST;
        } ();
        prob.add_sample(C {c.real(), c.imag()}, l);
    }

    using model_t = svm::model<kernel_t, label_t>;
    model_t model(std::move(prob), svm::parameters<kernel_t> {0.01});
    const size_t nr_labels = model.nr_labels();
    CHECK(nr_labels == N);
    const size_t nr_classifiers = model.nr_classifiers();
    CHECK(nr_classifiers == NC);

    std::vector<label_t> all_labels = {quadrants::NORTH_WEST,
                                       quadrants::SOUTH_EAST,
                                       quadrants::NORTH_EAST,
                                       quadrants::SOUTH_WEST};
    std::sort(all_labels.begin(), all_labels.end());

    auto labels = model.labels();
    static_assert(std::is_same<decltype(labels), std::array<label_t, N>>::value);
    CHECK(std::equal(all_labels.begin(), all_labels.end(), labels.begin()));

    auto classifiers = model.classifiers();
    static_assert(std::is_same<decltype(classifiers),
                               std::array<model_t::classifier_type, NC>>::value);
    auto check_consistency = [&] (size_t c, size_t i, size_t j) {
        CHECK(classifiers[c].labels().first  == labels[i]);
        CHECK(classifiers[c].labels().second == labels[j]);
        auto clss = model.classifier(labels[i], labels[j]);
        CHECK(clss.labels().first  == labels[i]);
        CHECK(clss.labels().second == labels[j]);
        auto clssr = model.classifier(labels[j], labels[i]);
        CHECK(clssr.labels().first  == labels[j]);
        CHECK(clssr.labels().second == labels[i]);
    };

    check_consistency(0, 0, 1);
    check_consistency(1, 0, 2);
    check_consistency(2, 0, 3);
    check_consistency(3, 1, 2);
    check_consistency(4, 1, 3);
    check_consistency(5, 2, 3);

    for (size_t i = 0; i < 25; ++i) {
        C c {uniform(rng), uniform(rng)};
        auto res = model(c);
        static_assert(std::is_same<decltype(res.second),
                                   std::array<double, NC>>::value);
        for (size_t i = 0; i < nr_classifiers; ++i) {
            auto cres = classifiers[i](c);
            CHECK(res.first == cres.first);
            CHECK(res.second[i] == cres.second);
            auto cresl = model.classifier(classifiers[i].labels().first,
                                          classifiers[i].labels().second)(c);
            CHECK(res.first == cresl.first);
            CHECK(res.second[i] == cresl.second);
            auto cresr = model.classifier(classifiers[i].labels().second,
                                          classifiers[i].labels().first)(c);
            CHECK(res.first == cresr.first);
            CHECK(res.second[i] == -cresr.second);
        }
    }

    auto rhos = model.rho();
    static_assert(std::is_same<decltype(rhos), std::array<double, NC>>::value);
    for (size_t i = 0; i < nr_classifiers; ++i) {
        double crho = classifiers[i].rho();
        CHECK(rhos[i] == crho);
    }
}

struct dynamic_label {
    static const size_t label_dim = 1;
    static const size_t nr_labels = svm::DYNAMIC;
    dynamic_label () : val(0) {}
    template <class Iterator,
              typename Tag = typename std::iterator_traits<Iterator>::value_type>
    dynamic_label (Iterator begin) : val (*begin) {}
    dynamic_label (double x) : val(x) {}
    operator double() const { return val; }
    double const * begin() const { return &val; }
    double const * end() const { return &val + 1; }
    friend bool operator== (dynamic_label lhs, dynamic_label rhs) {
        return lhs.val == rhs.val;
    }
    friend std::ostream & operator<< (std::ostream & os, dynamic_label l) {
        return os << l.val;
    }
private:
    double val;
};

static_assert(svm::traits::is_dynamic_label<dynamic_label>::value);

TEST_CASE("dynamic-multi-class") {
    using label_t = dynamic_label;
    using cmplx = std::complex<double>;
    using kernel_t = svm::kernel::linear;
    using problem_t = svm::problem<kernel_t, label_t>;
    using C = typename problem_t::input_container_type;

    const size_t M = 10000;
    const size_t N = 4;
    const size_t NC = 6;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);

    problem_t prob(2);
    for (size_t i = 0; i < M; ++i) {
        cmplx c {uniform(rng), uniform(rng)};
        label_t l(std::floor(std::arg(c) / (2 * M_PI) * N));
        prob.add_sample(C {c.real(), c.imag()}, l);
    }

    using model_t = svm::model<kernel_t, label_t>;
    model_t model(std::move(prob), svm::parameters<kernel_t> {0.01});
    const size_t nr_labels = model.nr_labels();
    CHECK(nr_labels == N);
    const size_t nr_classifiers = model.nr_classifiers();
    CHECK(nr_classifiers == NC);

    std::vector<label_t> all_labels(N);
    std::iota(all_labels.begin(), all_labels.end(), std::floor(-0.5 * N));

    auto labels = model.labels();
    static_assert(std::is_same<decltype(labels), std::vector<label_t>>::value);
    CHECK(std::equal(all_labels.begin(), all_labels.end(), labels.begin()));

    auto classifiers = model.classifiers();
    static_assert(std::is_same<decltype(classifiers),
                               std::vector<model_t::classifier_type>>::value);
    CHECK(classifiers.size() == nr_classifiers);

    for (size_t i = 0; i < 25; ++i) {
        C c {uniform(rng), uniform(rng)};
        auto res = model(c);
        static_assert(std::is_same<decltype(res.second), std::vector<double>>::value);
        CHECK(res.second.size() == nr_classifiers);
        for (size_t i = 0; i < nr_classifiers; ++i) {
            auto cres = classifiers[i](c);
            CHECK(res.first == cres.first);
            CHECK(res.second[i] == cres.second);
            auto cresl = model.classifier(classifiers[i].labels().first,
                                          classifiers[i].labels().second)(c);
            CHECK(res.first == cresl.first);
            CHECK(res.second[i] == cresl.second);
            auto cresr = model.classifier(classifiers[i].labels().second,
                                          classifiers[i].labels().first)(c);
            CHECK(res.first == cresr.first);
            CHECK(res.second[i] == -cresr.second);
        }
    }

    auto rhos = model.rho();
    static_assert(std::is_same<decltype(rhos), std::vector<double>>::value);
    CHECK(rhos.size() == nr_classifiers);
    for (size_t i = 0; i < nr_classifiers; ++i) {
        double crho = classifiers[i].rho();
        CHECK(rhos[i] == crho);
    }
}
