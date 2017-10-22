#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "svm.h"

#include <stdexcept>


namespace svm {

    template <class Kernel>
    class model {
    public:
        typedef problem<Kernel> problem_t;
        typedef parameters<Kernel> parameters_t;

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

    private:
        problem_t prob;
        svm_problem svm_prob;
        parameters_t params;
        struct svm_model * m;
    };

}
