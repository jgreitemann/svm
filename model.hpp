#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
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
            data_view support_vec () const {
                return data_view(*sv);
            }
            std::pair<double, data_view> operator* () const {
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
            const_iterator (double * yalpha, struct svm_node ** sv)
                : yalpha(yalpha), sv(sv) {}
            double * yalpha;
            struct svm_node ** sv;
        };

        model (problem_t&& problem, parameters_t const& parameters)
            : prob(std::move(problem)),
              params(parameters)
        {
            svm_prob = prob.generate();
            const char * err = svm_check_parameter(&svm_prob, params.svm_params_ptr());
            if (err) {
                std::string err_str(err);
                throw std::runtime_error(err_str);
            }
            m = svm_train(&svm_prob, params.svm_params_ptr());
        }

        ~model () noexcept {
            svm_free_and_destroy_model(&m); // WTF
        }

        template <typename Problem = problem_t, typename = typename std::enable_if<!Problem::is_precomputed>::type>
        double operator() (input_container_type const& xj) {
            return svm_predict(m, xj.ptr());
        }

        template <typename Problem = problem_t, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
        double operator() (input_container_type const& xj) {
            dataset kernelized = prob.kernelize(xj);
            return svm_predict(m, kernelized.ptr());
        }

        const_iterator begin () const {
            return const_iterator(m->sv_coef[0], m->SV);
        }
        const_iterator end () const {
            return const_iterator(m->sv_coef[0] + m->l, m->SV + m->l);
        }

    private:
        problem_t prob;
        svm_problem svm_prob;
        parameters_t params;
        struct svm_model * m;
    };

    template <class Kernel>
    struct introspective_model : public model<Kernel> {
        using model<Kernel>::model;
    };

}
