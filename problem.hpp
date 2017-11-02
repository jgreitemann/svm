#pragma once

#include "dataset.hpp"
#include "svm.h"

#include <type_traits>
#include <utility>
#include <vector>


namespace svm {

    namespace detail {

        template <class Container>
        class basic_problem {
        public:
            typedef Container input_container_type;

            basic_problem(size_t dim) : dimension(dim) {};
            basic_problem(basic_problem const&) = delete;
            basic_problem & operator= (basic_problem const&) = delete;
            basic_problem(basic_problem &&) = default;
            basic_problem & operator= (basic_problem &&) = default;

            void add_sample(Container && ds, double label) {
                orig_data.push_back(std::move(ds));
                labels.push_back(label);
            }

            void add_sample(Container const& ds, double label) {
                orig_data.push_back(ds);
                labels.push_back(label);
            }

            std::pair<Container const&, double> operator[] (size_t i) const {
                return std::pair<Container const&, double>(orig_data[i], labels[i]);
            }

            size_t size () const {
                return orig_data.size();
            }

            size_t dim () const {
                return dimension;
            }
        protected:
            std::vector<Container> orig_data;
            std::vector<double> labels;
        private:
            size_t dimension;
        };

        class patch_through_problem : public basic_problem<dataset> {
        public:
            static bool const is_precomputed = false;
            patch_through_problem(size_t dim) : basic_problem<dataset>(dim) {};
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

        template <class Kernel, class Container>
        class precompute_kernel_problem : public basic_problem<Container> {
        public:
            static bool const is_precomputed = true;
            precompute_kernel_problem(precompute_kernel_problem const&) = delete;
            precompute_kernel_problem & operator= (precompute_kernel_problem const&) = delete;
            precompute_kernel_problem(precompute_kernel_problem &&) = default;
            precompute_kernel_problem & operator= (precompute_kernel_problem &&) = default;

            template <typename = typename std::enable_if<std::is_default_constructible<Kernel>::value>::type>
            precompute_kernel_problem (size_t dim)
                : basic_problem<Container>(dim) {}

            precompute_kernel_problem (const Kernel & k, size_t dim)
                : basic_problem<Container>(dim), kernel(k) {}

            struct svm_problem generate() {
                kernel_data.clear();
                ptrs.clear();
                int i = 1;
                for (Container const& xi : orig_data) {
                    kernel_data.push_back(kernelize(xi, i));
                    ptrs.push_back(kernel_data.back().ptr());
                    ++i;
                }
                struct svm_problem p;
                p.x = ptrs.data();
                p.y = labels.data();
                p.l = labels.size();
                return p;
            }
            dataset kernelize(Container const& xi, double index = 1) {
                std::vector<double> v;
                v.push_back(index);
                for (Container const& xj : orig_data) {
                    v.push_back(kernel(xi, xj));
                }
                return dataset(v, 0, false);
            }
        private:
            using basic_problem<Container>::orig_data;
            using basic_problem<Container>::labels;
            Kernel kernel;
            std::vector<dataset> kernel_data;
            std::vector<struct svm_node *> ptrs;
        };

    }

    template <class Kernel>
    class problem : public detail::precompute_kernel_problem<Kernel, typename Kernel::input_container_type> {
        using detail::precompute_kernel_problem<Kernel, typename Kernel::input_container_type>::precompute_kernel_problem;
    };

}
