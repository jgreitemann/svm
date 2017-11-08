#pragma once

#include "svm.h"

#include <stdexcept>


namespace svm {

    namespace detail {

        class basic_parameters {
        public:
            basic_parameters (struct svm_parameter const& p) : params(p) {
                if (p.svm_type != NU_SVC) {
                    std::invalid_argument("SVMs other than nu-SVC are not "
                                          "currently supported");
                }
            };
            basic_parameters (double nu = 0.1) {
                params.svm_type = NU_SVC;
                params.cache_size = 100;
                params.eps = 1e-3;
                params.nu = 0.1;
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
