#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "svm.h"

#include <stdexcept>


namespace svm {

    namespace kernel {

        struct linear;

    }

    template <>
    class problem<kernel::linear> : public detail::patch_through_problem {
        using detail::patch_through_problem::patch_through_problem;
    };

    template <>
    class parameters<kernel::linear> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != LINEAR) {
                std::invalid_argument("parameters do not use linear kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = LINEAR;
        }
    };


}
