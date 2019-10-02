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

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <svm/dataset.hpp>

#include <svm/libsvm/svm.h>

#include <svm/serialization/serializer.hpp>

#include <svm/traits/label_traits.hpp>


namespace svm {
namespace serialization {

    struct ascii_tag;

    template <typename Model>
    struct model_serializer<ascii_tag, Model> {

        model_serializer (Model & m)
            : model_(m), prob_serializer(m.prob, !problem_t::is_precomputed) {}

        void save (std::string const& filename) const {
            prob_serializer.save(filename + ".prob");
            if (svm_save_model((filename + ".model").c_str(), model_.m) != 0) {
                throw std::runtime_error("Failed to save model to file "
                                            + filename + ".model");
            }
        }

        void load (std::string const& filename) {
            prob_serializer.load(filename + ".prob");
            if (model_.m)
                svm_free_and_destroy_model(&model_.m);
            model_.m = svm_load_model((filename + ".model").c_str());
            if (!model_.m) {
                throw std::runtime_error("Failed to load model from file "
                                            + filename + ".model");
            }
            model_.params_ = typename Model::parameters_t(model_.m->param);
            model_.init_perm();
        }

    private:
        using problem_t = typename Model::problem_t;
        Model & model_;
        problem_serializer<ascii_tag, problem_t> prob_serializer;
    };

    template <class Problem>
    struct problem_serializer<ascii_tag, Problem> {

        problem_serializer (Problem & prob, bool skip_samples = false)
            : prob_(prob), full(!skip_samples) {}

        void save (std::string const& filename) const {
            using input_t = typename Problem::input_container_type;
            using label_t = typename Problem::label_type;
            using ltraits = typename::svm::traits::label_traits<label_t>;
            using view_t = typename std::conditional<
                std::is_same<svm::dataset, input_t>::value,
                svm::data_view, input_t const&>::type;

            std::ofstream os(filename);
            os << prob_.dim() << '\n';

            if (full) {
                for (size_t i = 0; i < prob_.size(); ++i) {
                    auto p = prob_[i];
                    view_t xs = p.first;
                    label_t const& l = p.second;
                    for (auto it = ltraits::begin(l); it != ltraits::end(l); ++it) {
                        os << *it << ' ';
                    }
                    os << '\t';
                    size_t j = 0;
                    for (double const& x : xs) {
                        os << ' ' << x;
                        ++j;
                    }
                    for (; j < prob_.dim(); ++j)
                        os << ' ' << 0.;
                    os << '\n';
                }
            }
        }

        void load (std::string const& filename) const {
            using input_t = typename Problem::input_container_type;
            using label_t = typename Problem::label_type;
            using ltraits = typename::svm::traits::label_traits<label_t>;

            std::ifstream is(filename);

            size_t dim;
            is >> dim;
            Problem prob(dim);

            if (full) {
                double ys[ltraits::label_dim];
                std::vector<double> xs(dim);
                while (is >> ys[0]) {
                    for (size_t j = 1; j < ltraits::label_dim; ++j) {
                        if (!(is >> ys[j])) {
                            throw std::runtime_error("incomplete problem");
                        }
                    }
                    for (size_t j = 0; j < dim; ++j) {
                        if (!(is >> xs[j])) {
                            throw std::runtime_error("incomplete problem");
                        }
                    }
                    prob.add_sample(input_t(xs.begin(), xs.end()),
                                    ltraits::from_iterator(std::begin(ys)));
                }
            }
            prob_ = std::move(prob);
        }

    private:
        Problem & prob_;
        bool full;
    };

}
    using serialization::ascii_tag;
}
