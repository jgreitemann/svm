/*   Support Vector Machine Library Wrappers
 *   Copyright (C) 2018  Jonas Greitemann
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

#include "doctest.h"
#include "svm-wrapper.hpp"
#include "label.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>


using svm::detail::basic_problem;

SVM_LABEL_BEGIN(quadrants, 4)
SVM_LABEL_ADD(NORTH_EAST)
SVM_LABEL_ADD(NORTH_WEST)
SVM_LABEL_ADD(SOUTH_WEST)
SVM_LABEL_ADD(SOUTH_EAST)
SVM_LABEL_END()

static_assert(!svm::traits::is_dynamic_label<quadrants::label>::value);

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
