#pragma once

#include "binomial.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "model.hpp"
#include "svm.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <boost/multi_array.hpp>


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
        double gamma () const { return params.gamma; }
        double coef0 () const { return params.coef0; }
    };


    template <size_t K>
    using tensor_t = boost::multi_array<double, K>;

    namespace detail {
        template <size_t D, size_t K>
        struct tensor_sequence : public tensor_sequence<D, K-1> {
            tensor_sequence (model<kernel::polynomial<D>> const& model)
                : tensor_sequence<D, K-1>(model)
            {
                double yalpha;
                data_view x;
                using index_t = typename tensor_t<K>::index;
                std::array<index_t, K> ind;
                for (index_t & z : ind)
                    z = 1;
                ind[0] = pow(N, K);
                t.resize(ind);
                for (index_t & z : ind)
                    z = 0;
                for (auto p : model) {
                    std::tie(yalpha, x) = std::move(p);
                    std::vector<double> xv(x.begin(), x.end());
                    xv.resize(N);
                    for (ind[0] = 0; ind[0] < t.shape()[0]; ++ind[0]) {
                        index_t m = ind[0];
                        double prod = 1.;
                        for (size_t k = 0; k < K; ++k, m /= N)
                            prod *= xv[m % N];
                        t(ind) += yalpha * prod;
                    }
                }
                double fac = binomial(D, K) * pow(model.params().gamma(), K)
                    * pow(model.params().coef0(), D-K);
                for (ind[0] = 0; ind[0] < t.shape()[0]; ++ind[0])
                    t(ind) *= fac;
                for (index_t & i : ind)
                    i = N;
                t.reshape(ind);
            }

            using tensor_sequence<D,0>::N;
        public:
            tensor_t<K> t;
        };

        template <size_t D>
        struct tensor_sequence<D,0> {
            tensor_sequence<D,0> (model<kernel::polynomial<D>> const& model)
                : N(model.dim()) {}

            const size_t N;
        };
    }

    template <size_t D>
    struct introspective_model<kernel::polynomial<D>> : public model<kernel::polynomial<D>> {

        using poly_model = model<kernel::polynomial<D>>;

    public:
        template <typename... Args>
        introspective_model (Args... args)
            : poly_model(std::forward<Args>(args)...),
              tseq(*this) {}

        template <size_t K, typename = typename std::enable_if<(K <= D && K > 0)>::type>
        tensor_t<K> const& tensor () const {
            static_assert(K <= D, "invalid tensor rank");
            return tseq.detail::tensor_sequence<D,K>::t;
        }

        template <size_t K, typename = typename std::enable_if<K == 0>::type>
        double tensor () const {
            return pow(poly_model::params().coef0(), D);
        }
        
    private:
        detail::tensor_sequence<D,D> tseq;
    };

}
