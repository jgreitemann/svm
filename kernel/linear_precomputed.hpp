#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "svm.h"

#include <stdexcept>


namespace svm {

    namespace kernel {

        struct linear_precomputed {
            double operator() (data_view xi, data_view xj) const {
                return dot(xi, xj);
            }
        };

    }

    template <>
    class problem<kernel::linear_precomputed>
        : public detail::precompute_kernel_problem<kernel::linear_precomputed>
    {
        using precompute_kernel_problem<kernel::linear_precomputed>::precompute_kernel_problem;
    };

    template <>
    class parameters<kernel::linear_precomputed> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != PRECOMPUTED) {
                std::invalid_argument("parameters do not use precomputed kernel");
            }
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = PRECOMPUTED;
        }
    };


}
