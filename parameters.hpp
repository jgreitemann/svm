#pragma once

#include "svm.h"

#include <stdexcept>


namespace svm {

    enum class machine_type {
        C_SVC = C_SVC,
        NU_SVC = NU_SVC,
        ONE_CLASS = ONE_CLASS,
        EPSILON_SVR = EPSILON_SVR,
        NU_SVR = NU_SVR
    };

    namespace detail {

        class basic_parameters {
        public:
            basic_parameters (struct svm_parameter const& p) : params(p) {}

            basic_parameters (double reg = 0.1, machine_type mtype = machine_type::NU_SVC) {
                switch (mtype) {
                case machine_type::C_SVC:
                    params.C = reg;
                    break;
                case machine_type::NU_SVC:
                    params.nu = reg;
                    break;
                default:
                    throw std::logic_error("SVM type not supported");
                }
                params.svm_type = static_cast<int>(mtype);
                params.cache_size = 100;
                params.eps = 1e-3;
                params.nr_weight = 0;
                params.weight_label = NULL;
                params.weight = NULL;
                params.shrinking = 1;
                params.probability = 0;
            }

            struct svm_parameter * svm_params_ptr () {
                return &params;
            }
        protected:
            struct svm_parameter params;
        };

    }
    
    template <class Kernel>
    class parameters;

}
