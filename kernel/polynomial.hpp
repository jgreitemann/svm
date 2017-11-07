#pragma once

#include "binomial.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>


namespace svm {

    namespace kernel {

        template <size_t D>
        struct polynomial {
            static size_t const Degree = D;
        };

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
        double gamma () const { return params.gamma; }
        double coef0 () const { return params.coef0; }
    };


    template <class Kernel, size_t K, size_t D = Kernel::Degree>
    struct tensor_introspector {
        static_assert(K <= D, "invalid tensor rank");

        using poly_model = model<Kernel>;

        tensor_introspector (poly_model const& model)
            : model_(model), N(model.dim())
        {
            fac = binomial(D, K) * pow(model.params().gamma(), K)
                * pow(model.params().coef0(), D-K);
        }

        template <size_t L=K, typename = typename std::enable_if<L != 0>::type>
        double tensor (std::array<size_t, K> ind) const {
            std::sort(ind.begin(), ind.end());
            double yalpha;
            data_view x;
            double sum = 0;
            for (auto p : model_) {
                std::tie(yalpha, x) = std::move(p);
                double prod = 1.;
                size_t j = 0;
                auto itX = x.begin();
                for (size_t i : ind) {
                    std::advance(itX, i - j);
                    j = i;
                    prod *= *itX;
                }
                sum += yalpha * prod;
            }
            return fac * sum;
        }

        template <size_t L=K, typename = typename std::enable_if<L == 0>::type>
        double tensor () const {
            return fac;
        }
    private:
        poly_model const& model_;
        size_t const N;
        double fac;
    };

}
