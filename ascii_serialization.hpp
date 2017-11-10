#pragma once

#include "serializer.hpp"
#include "svm.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace svm {

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
            using view_t = typename std::conditional<
                std::is_same<svm::dataset, input_t>::value,
                svm::data_view, input_t const&>::type;

            std::ofstream os(filename);
            os << prob_.dim() << '\n';

            if (full) {
                for (size_t i = 0; i < prob_.size(); ++i) {
                    auto p = prob_[i];
                    view_t xs = p.first;
                    size_t j = 0;
                    os << p.second;
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

            std::ifstream is(filename);

            size_t dim;
            is >> dim;
            Problem prob(dim);

            if (full) {
                double y;
                std::vector<double> xs(dim);
                while (is >> y) {
                    for (size_t j = 0; j < dim; ++j) {
                        if (!(is >> xs[j])) {
                            throw std::runtime_error("incomplete problem");
                        }
                    }
                    prob.add_sample(input_t(xs.begin(), xs.end()), y);
                }
            }
            prob_ = std::move(prob);
        }

    private:
        Problem & prob_;
        bool full;
    };

}
