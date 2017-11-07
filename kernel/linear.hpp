#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <algorithm>
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

    template <class Kernel>
    struct linear_introspector {
        using model_t = model<Kernel>;

        linear_introspector(model_t const& model) : model_(model) {}
        
        double coefficient(size_t i) const {
            double c = 0;
            for (auto p : model_) {
                double yalpha = p.first;
                auto const& x = p.second;
                auto itX = x.begin();
                std::advance(itX, i);
                c += yalpha * *itX;
            }
            return c;
        }

    private:
        model_t const& model_;
    };

}
