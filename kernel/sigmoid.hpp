#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <stdexcept>


namespace svm {

    namespace kernel {

        struct sigmoid {
        };

    }

    template <>
    class problem<kernel::sigmoid> : public detail::patch_through_problem {
        using detail::patch_through_problem::patch_through_problem;
    };

    template <>
    class parameters<kernel::sigmoid> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != SIGMOID) {
                std::invalid_argument("parameters do not use radial basis "
                                      "function (SIGMOID) kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = RBF;
            params.gamma = 1.;
            params.coef0 = 0.;
        }
        double const& gamma () const { return params.gamma; }
        double & gamma () { return params.gamma; }
        double const& coef0 () const { return params.coef0; }
        double & coef0 () { return params.coef0; }
    };

}
