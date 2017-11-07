#pragma once

#include "dataset.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "svm.h"

#include <algorithm>
#include <vector>
#include <stdexcept>


namespace svm {

    namespace kernel {

        struct linear_precomputed {
            typedef std::vector<double> input_container_type;
            double operator() (input_container_type const& xi,
                               input_container_type const& xj) const {
                auto iti = xi.begin();
                auto itj = xj.begin();
                double sum = 0;
                for (; iti != xi.end() && itj != xj.end(); ++iti, ++itj) {
                    sum += *iti * *itj;
                }
                return sum;
            }
        };

    }

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
