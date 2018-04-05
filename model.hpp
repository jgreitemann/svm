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

#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "serializer.hpp"
#include "svm.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>


namespace svm {

    template <class Kernel, class Label = double>
    class model {
    public:
        typedef problem<Kernel, Label> problem_t;
        typedef parameters<Kernel> parameters_t;
        typedef typename problem_t::input_container_type input_container_type;
        typedef Label label_type;
        typedef std::pair<Label, Label> classifier_type;

        static const size_t nr_labels = detail::label_traits<Label>::nr_labels;
        static const size_t nr_classifiers = nr_labels * (nr_labels - 1) / 2;

        using decision_type = std::conditional_t<nr_classifiers == 1,
                                                 double,
                                                 std::array<double, nr_classifiers>>;

        using label_arr_t = std::array<label_type, nr_labels>;
        using classifier_arr_t = std::array<classifier_type, nr_classifiers>;

        class const_iterator {
        public:
            const_iterator & operator++ () {
                ++yalpha;
                ++sv;
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () {
                --yalpha;
                --sv;
                return *this;
            }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            double coef () const {
                return *yalpha;
            }

            template <typename Problem = problem_t,
                      typename = std::enable_if_t<!Problem::is_precomputed>>
            data_view support_vec () const {
                return data_view(*sv);
            }

            template <typename Problem = problem_t,
                      typename = std::enable_if_t<Problem::is_precomputed>,
                      bool dummy = false>
            input_container_type const& support_vec () const {
                data_view permutation_index(*sv, 0);
                return prob[permutation_index.front()-1].first;
            }

            auto operator* () const {
                return std::make_pair(coef(), support_vec());
            }

            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return lhs.yalpha == rhs.yalpha && lhs.sv == rhs.sv;
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) {
                return lhs.yalpha != rhs.yalpha || lhs.sv != rhs.sv;
            }

            friend model;
        private:
            const_iterator (double * yalpha, struct svm_node ** sv, problem_t const& prob)
                : yalpha(yalpha), sv(sv), prob(prob) {}
            double * yalpha;
            struct svm_node ** sv;
            problem_t const& prob;
        };

        model () : prob(0), m(nullptr) {}

        model (problem_t && problem, parameters_t const& parameters)
            : prob(std::move(problem)),
              params_(parameters)
        {
            struct svm_problem svm_prob = prob.generate();
            const char * err = svm_check_parameter(&svm_prob, params_.svm_params_ptr());
            if (err) {
                std::string err_str(err);
                throw std::runtime_error(err_str);
            }
            m = svm_train(&svm_prob, params_.svm_params_ptr());
            if (std::isnan(rho()))
                throw std::runtime_error("SVM returned NaN. Specified nu is infeasible.");
            if (m->nr_class != nr_labels)
                throw std::runtime_error("inconsistent number of label values");
        }

        model (model const&) = delete;
        model & operator= (model const&) = delete;

        model (model && other)
            : prob(std::move(other.prob)),
              params_(other.params_),
              m(other.m)
        {
            other.m = nullptr;
        }

        model & operator= (model && other) {
            prob = std::move(other.prob);
            params_ = std::move(other.params_);
            if (m)
                svm_free_and_destroy_model(&m);
            m = other.m;
            other.m = nullptr;
            return *this;
        }

        ~model () noexcept {
            if (m)
                svm_free_and_destroy_model(&m); // WTF
        }

        label_arr_t labels () const {
            label_arr_t ret;
            std::copy(m->label, m->label + nr_labels, ret);
            return ret;
        }

        classifier_arr_t classifiers () const {
            auto ls = labels();
            classifier_arr_t ret;
            auto it = ret.begin();
            for (auto l1 = ls.begin(); l1 != ls.end(); ++l1)
                for (auto l2 = l1; l2 != ls.end(); ++l2, ++it)
                    *it = classifier_type(*l1, *l2);
        }

        template <typename Problem = problem_t,
                  typename = std::enable_if_t<!Problem::is_precomputed>>
        std::pair<Label, decision_type> operator() (input_container_type const& xj) {
            decision_type dec;
            Label label(svm_predict_values(m, xj.ptr(), reinterpret_cast<double*>(&dec)));
            return std::make_pair(label, dec);
        }

        template <typename Problem = problem_t,
                  typename = std::enable_if_t<Problem::is_precomputed>,
                  bool dummy = false>
        std::pair<Label, decision_type> operator() (input_container_type const& xj) {
            dataset kernelized = prob.kernelize(xj);
            decision_type dec;
            Label label(svm_predict_values(m, kernelized.ptr(), reinterpret_cast<double*>(&dec)));
            return std::make_pair(label, dec);
        }

        const_iterator begin () const {
            return const_iterator(m->sv_coef[0], m->SV, prob);
        }
        const_iterator end () const {
            return const_iterator(m->sv_coef[0] + m->l, m->SV + m->l, prob);
        }

        double rho () const {
            return m->rho[0];
        }

        std::pair<size_t, size_t> nSV () const {
            return {m->nSV[0], m->nSV[1]};
        }

        size_t dim () const {
            return prob.dim();
        }

        parameters_t const& params () const {
            return params_;
        }

        template <typename Tag, typename Model>
        friend struct model_serializer;

    private:
        problem_t prob;
        parameters_t params_;
        struct svm_model * m;
    };

}
