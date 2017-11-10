#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "serializer.hpp"
#include "svm.h"

#include <utility>
#include <stdexcept>
#include <type_traits>


namespace svm {

    template <class Kernel>
    class model {
    public:
        typedef problem<Kernel> problem_t;
        typedef parameters<Kernel> parameters_t;
        typedef typename problem_t::input_container_type input_container_type;

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

            template <typename Problem = problem_t, typename = typename std::enable_if<!Problem::is_precomputed>::type>
            data_view support_vec () const {
                return data_view(*sv);
            }

            template <typename Problem = problem_t, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
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

        template <typename Problem = problem_t, typename = typename std::enable_if<!Problem::is_precomputed>::type>
        std::pair<double, double> operator() (input_container_type const& xj) {
            double dec;
            double label =  svm_predict_values(m, xj.ptr(), &dec);
            return std::make_pair(label, dec);
        }

        template <typename Problem = problem_t, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
        std::pair<double, double> operator() (input_container_type const& xj) {
            dataset kernelized = prob.kernelize(xj);
            double dec;
            double label = svm_predict_values(m, kernelized.ptr(), &dec);
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
