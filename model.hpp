#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "svm.h"

#include <stdexcept>
#include <type_traits>


namespace svm {

    // forward declaration
    template <class Kernel>
    class model;

    template <class Kernel>
    struct introspection_policy {
        introspection_policy (model<Kernel> const&) {}
    };

    template <class Kernel>
    class model : public introspection_policy<Kernel> {
    public:
        typedef problem<Kernel> problem_t;
        typedef parameters<Kernel> parameters_t;

        model (problem_t&& problem, parameters_t const& parameters)
            : introspection_policy<Kernel>(*this),
              prob(std::move(problem)),
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

        template <typename Problem = problem_t, typename = typename std::enable_if<!Problem::is_precomputed>::type>
        double operator() (dataset const& xj) {
            return svm_predict(m, xj.ptr());
        }

        template <typename Problem = problem_t, typename = typename std::enable_if<Problem::is_precomputed>::type, bool dummy = false>
        double operator() (dataset const& xj) {
            dataset kernelized = prob.kernelize(xj);
            return svm_predict(m, kernelized.ptr());
        }

        ~model () noexcept {
            svm_free_and_destroy_model(&m); // WTF
        }

    private:
        problem_t prob;
        svm_problem svm_prob;
        parameters_t params;
        struct svm_model * m;
    };

}
