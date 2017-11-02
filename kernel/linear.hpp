#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <stdexcept>
#include <utility>


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

    template <>
    struct introspective_model<kernel::linear> : public model<kernel::linear> {
        template <typename... Args>
        introspective_model (Args... args)
            : model<kernel::linear>(std::forward<Args>(args)...), C(dim(), 0.)
        {
            double yalpha;
            data_view x;
            for (auto p : *this) {
                std::tie(yalpha, x) = std::move(p);
                auto itC = C.begin();
                for (double xj : x) {
                    *itC += yalpha * xj;
                    ++itC;
                }
            }

        }
        
        std::vector<double> const& coefficients() const { return C; }
    private:
        std::vector<double> C;
    };

}
