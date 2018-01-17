#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <stdexcept>


namespace svm {

    namespace kernel {

        struct rbf {
        };

    }

    template <>
    class problem<kernel::rbf> : public detail::patch_through_problem {
        using detail::patch_through_problem::patch_through_problem;
    };

    template <>
    class parameters<kernel::rbf> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != RBF) {
                std::invalid_argument("parameters do not use radial basis "
                                      "function (RBF) kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = RBF;
            params.gamma = 1.;
        }
        double const& gamma () const { return params.gamma; }
        double & gamma () { return params.gamma; }
    };

}
