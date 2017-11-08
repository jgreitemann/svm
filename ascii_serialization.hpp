#pragma once

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
    struct serializer<ascii_tag, Model> {

        serializer (Model & m) : model_(m) {}

        void save (std::string const& filename) const {
            save_problem(model_.prob, filename + ".prob");
            if (svm_save_model((filename + ".model").c_str(), model_.m) != 0) {
                throw std::runtime_error("Failed to save model to file "
                                            + filename + ".model");
            }
        }

        void load (std::string const& filename) {
            model_.prob = load_problem<typename Model::problem_t>(filename + ".prob");
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

        template <class Problem, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
        void save_problem (Problem const& prob, std::string const& filename) const {
            using input_t = typename Problem::input_container_type;

            std::ofstream os(filename);
            os << prob.dim() << '\n';

            for (size_t i = 0; i < prob.size(); ++i) {
                auto p = prob[i];
                input_t const& xs = p.first;
                size_t j = 0;
                os << p.second;
                for (double const& x : xs) {
                    os << ' ' << x;
                    ++j;
                }
                for (; j < prob.dim(); ++j)
                    os << ' ' << 0.;
                os << '\n';
            }
        }

        template <class Problem, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
        Problem load_problem (std::string const& filename) const {
            using input_t = typename Problem::input_container_type;

            std::ifstream is(filename);

            size_t dim;
            is >> dim;
            Problem prob(dim);

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
            return prob;
        }

        template <class Problem, typename = typename std::enable_if<!Problem::is_precomputed>::type>
        void save_problem(Problem const& prob, std::string const& filename) const {
            std::ofstream os(filename);
            os << prob.dim() << '\n';
        }

        template <class Problem, typename = typename std::enable_if<!Problem::is_precomputed>::type>
        Problem load_problem (std::string const& filename) const {
            std::ifstream is(filename);

            size_t dim;
            is >> dim;
            Problem prob(dim);
            return prob;
        }

        Model & model_;

    };

}
