#pragma once

#include "dataset.hpp"
#include "svm.h"

#include <vector>
#include <type_traits>


namespace svm {

    namespace detail {

        class basic_problem {
        public:
            basic_problem() = default;
            basic_problem(basic_problem const&) = delete;
            basic_problem & operator= (basic_problem const&) = delete;
            basic_problem(basic_problem &&) = default;
            basic_problem & operator= (basic_problem &&) = default;

            void add_sample(dataset&& ds, double label) {
                orig_data.push_back(std::move(ds));
                labels.push_back(label);
            }
        protected:
            std::vector<dataset> orig_data;
            std::vector<double> labels;
        };

        class patch_through_problem : public basic_problem {
        public:
            static bool const is_precomputed = false;
            patch_through_problem() = default;
            patch_through_problem(patch_through_problem const&) = delete;
            patch_through_problem & operator= (patch_through_problem const&) = delete;
            patch_through_problem(patch_through_problem &&) = default;
            patch_through_problem & operator= (patch_through_problem &&) = default;

            struct svm_problem generate() {
                ptrs.clear();
                for (dataset & ds : orig_data)
                    ptrs.push_back(ds.ptr());
                struct svm_problem p;
                p.x = ptrs.data();
                p.y = labels.data();
                p.l = labels.size();
                return p;
            }
        private:
            std::vector<struct svm_node *> ptrs;
        };

        template <class Kernel>
        class precompute_kernel_problem : public basic_problem {
        public:
            static bool const is_precomputed = true;
            precompute_kernel_problem(precompute_kernel_problem const&) = delete;
            precompute_kernel_problem & operator= (precompute_kernel_problem const&) = delete;
            precompute_kernel_problem(precompute_kernel_problem &&) = default;
            precompute_kernel_problem & operator= (precompute_kernel_problem &&) = default;

            template <typename = typename std::enable_if<std::is_default_constructible<Kernel>::value>::type>
            precompute_kernel_problem () {}

            precompute_kernel_problem (const Kernel & k) : kernel(k) {}

            struct svm_problem generate() {
                kernel_data.clear();
                ptrs.clear();
                int i = 1;
                for (dataset & di : orig_data) {
                    kernel_data.push_back(kernelize(di, i));
                    ptrs.push_back(kernel_data.back().ptr());
                    ++i;
                }
                struct svm_problem p;
                p.x = ptrs.data();
                p.y = labels.data();
                p.l = labels.size();
                return p;
            }
            dataset kernelize(dataset const& di, double index = 1) {
                std::vector<double> v;
                v.push_back(index);
                for (dataset const& dj : orig_data) {
                    v.push_back(kernel(di, dj));
                }
                return dataset(v);
            }
        private:
            Kernel kernel;
            std::vector<dataset> kernel_data;
            std::vector<struct svm_node *> ptrs;
        };

    }

    template <class Kernel>
    class problem : public detail::precompute_kernel_problem<Kernel> {
        using detail::precompute_kernel_problem<Kernel>::precompute_kernel_problem;
    };

}
