#pragma once

#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <stdexcept>
#include <utility>


namespace svm {

    namespace kernel {

        template <size_t D>
        struct polynomial;

    }

    template <size_t D>
    class problem<kernel::polynomial<D>> : public detail::patch_through_problem {
        using detail::patch_through_problem::patch_through_problem;
    };

    template <size_t D>
    class parameters<kernel::polynomial<D>> : public detail::basic_parameters {
    public:
        parameters (struct svm_parameter const& p) : detail::basic_parameters(p) {
            if (p.kernel_type != POLY) {
                std::invalid_argument("parameters do not use polynomial kernel");
            }
        }
        template <typename... Args>
        parameters (double gamma, double c0, Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = POLY;
            params.degree = D;
            params.gamma = gamma;
            params.coef0 = c0;
        }
        template <typename... Args>
        parameters (Args... args) : detail::basic_parameters(args...) {
            params.kernel_type = POLY;
            params.degree = D;
            params.gamma = 1.;
            params.coef0 = 0.;
        }
    };

    // template <>
    // struct introspective_model<kernel::linear> : public model<kernel::linear> {
    //     template <typename... Args>
    //     introspective_model (Args... args)
    //         : model<kernel::linear>(std::forward<Args>(args)...)
    //     {
    //         double yalpha;
    //         data_view x;
    //         for (auto p : *this) {
    //             std::tie(yalpha, x) = std::move(p);
    //             auto itC = C.begin();
    //             for (double xj : x) {
    //                 if (itC == C.end()) {
    //                     C.push_back(0);
    //                     itC = --C.end();
    //                 }
    //                 *itC += yalpha * xj;
    //                 ++itC;
    //             }
    //         }

    //     }
        
    //     std::vector<double> const& coefficients() const { return C; }
    // private:
    //     std::vector<double> C;
    // };

}
