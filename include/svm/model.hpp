/*   Support Vector Machine Library Wrappers
 *   Copyright (C) 2018-2019  Jonas Greitemann
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program, see the file entitled "LICENCE" in the
 *   repository's root directory, or see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <svm/dataset.hpp>
#include <svm/problem.hpp>
#include <svm/parameters.hpp>
#include <svm/detail/container_factory.hpp>
#include <svm/libsvm/svm.h>
#include <svm/serialization/serializer.hpp>
#include <svm/traits/label_traits.hpp>


namespace svm {

    template <class Kernel, class Label = double>
    class model {
    private:
        static const size_t NRL = traits::label_traits<Label>::nr_labels;
        static const size_t NRC = NRL * (NRL - 1) / 2;
    public:
        typedef Kernel kernel_type;
        typedef problem<Kernel, Label> problem_t;
        typedef parameters<Kernel> parameters_t;
        typedef typename problem_t::input_container_type input_container_type;
        typedef Label label_type;

        struct classifier_type {

            using kernel_type = model::kernel_type;

            struct const_iterator {
                using support_vec_type = std::conditional_t<problem_t::is_precomputed,
                                                            input_container_type,
                                                            data_view>;
                using support_vec_ref = std::conditional_t<problem_t::is_precomputed,
                                                           input_container_type const&,
                                                           data_view>;
                using value_type = std::pair<double, support_vec_type>;
                using difference_type = size_t;
                using reference = std::pair<double const&, support_vec_ref> const;
                using pointer = std::pair<double, support_vec_ref> const *;
                using iterator_category = std::bidirectional_iterator_tag;

                const_iterator & operator++ () {
                    ++yalpha;
                    ++sv;
                    if (yalpha == yalpha_1_end) {
                        yalpha = yalpha_2;
                        sv = sv_2;
                    }
                    return *this;
                }
                const_iterator operator++ (int) {
                    const_iterator old(*this);
                    ++(*this);
                    return old;
                }
                const_iterator & operator-- () {
                    if (yalpha == yalpha_2) {
                        yalpha = yalpha_1_end;
                        sv = sv_1_end;
                    }
                    --yalpha;
                    --sv;
                    return *this;
                }
                const_iterator operator-- (int) {
                    const_iterator old(*this);
                    --(*this);
                    return old;
                }
                double coef () const {
                    return *yalpha * swapped;
                }

                template <typename Problem = problem_t,
                        typename = std::enable_if_t<!Problem::is_precomputed>>
                data_view support_vec () const {
                    return data_view(*sv);
                }

                template <typename Problem = problem_t,
                        typename = std::enable_if_t<Problem::is_precomputed>,
                        bool dummy = false>
                input_container_type const& support_vec () const {
                    data_view permutation_index(*sv, 0);
                    return prob[permutation_index.front()-1].first;
                }

                auto operator* () const {
                    return std::make_pair(coef(), support_vec());
                }

                friend bool operator== (const_iterator lhs, const_iterator rhs) {
                    return lhs.yalpha == rhs.yalpha && lhs.sv == rhs.sv;
                }
                friend bool operator!= (const_iterator lhs, const_iterator rhs) {
                    return lhs.yalpha != rhs.yalpha || lhs.sv != rhs.sv;
                }

                friend struct classifier_type;

            private:
                const_iterator (double * yalpha_1, struct svm_node ** sv_1,
                                double * yalpha_1_end, struct svm_node ** sv_1_end,
                                double * yalpha_2, struct svm_node ** sv_2,
                                int swapped,
                                problem_t const& prob)
                    : yalpha(yalpha_1)
                    , yalpha_1_end(yalpha_1_end)
                    , yalpha_2(yalpha_2)
                    , sv(sv_1)
                    , sv_1_end(sv_1_end)
                    , sv_2(sv_2)
                    , swapped(swapped)
                    , prob(prob)
                {}
                double * yalpha, * yalpha_1_end, * yalpha_2;
                struct svm_node ** sv, **sv_1_end, **sv_2;
                int swapped;
                problem_t const& prob;
            };

            using iterator = const_iterator;

            classifier_type (model const& parent, size_t k1, size_t k2)
                : parent(parent), k1(k1), k2(k2) {
                if (k1 > k2) {
                    swapped = -1;
                    std::swap(k1, k2);
                    std::swap(this->k1, this->k2);
                } else {
                    swapped = 1;
                }
                size_t sum = 0;
                for (size_t k = 0; k < parent.nr_labels(); ++k) {
                    if (k == k1)
                        k1_offset = sum;
                    if (k == k2)
                        k2_offset = sum;
                    sum += parent.m->nSV[k];
                }
                k_comb = k1 * (parent.nr_labels() - 1) - k1 * (k1 - 1) / 2 + k2 - k1 - 1;
            }

            const_iterator begin () const {
                return const_iterator {
                    parent.m->sv_coef[k2-1] + k1_offset,
                    parent.m->SV + k1_offset,
                    parent.m->sv_coef[k2-1] + k1_offset + parent.m->nSV[k1],
                    parent.m->SV + k1_offset + parent.m->nSV[k1],
                    parent.m->sv_coef[k1] + k2_offset,
                    parent.m->SV + k2_offset,
                    swapped,
                    parent.prob
                };
            }

            const_iterator end () const {
                return const_iterator {
                    parent.m->sv_coef[k1] + k2_offset + parent.m->nSV[k2],
                    parent.m->SV + k2_offset + parent.m->nSV[k2],
                    parent.m->sv_coef[k2-1] + k1_offset + parent.m->nSV[k1],
                    parent.m->SV + k1_offset + parent.m->nSV[k1],
                    parent.m->sv_coef[k1] + k2_offset,
                    parent.m->SV + k2_offset,
                    swapped,
                    parent.prob
                };
            }

            template <typename...,
                      typename L = Label,
                      typename = std::enable_if_t<traits::is_binary_label<L>::value>>
            std::pair<Label, double> operator() (input_container_type const& xj) {
                auto p = parent.raw_eval(xj);
                double & dec = detail::container_factory<decltype(p.second)>::ptr(p.second)[k_comb];
                dec *= swapped;
                return p;
            }

            template <typename...,
                      typename L = Label,
                      typename = std::enable_if_t<!traits::is_binary_label<L>::value>,
                      bool dummy = true>
            std::pair<Label, double> operator() (input_container_type const& xj) {
                auto p = parent.raw_eval(xj);
                double & dec = detail::container_factory<decltype(p.second)>::ptr(p.second)[k_comb];
                dec *= swapped;
                return {p.first, dec};
            }

            double rho () const {
                return swapped * parent.m->rho[k_comb];
            }

            parameters_t const& params () const {
                return parent.params();
            }

            std::pair<Label,Label> labels() const {
                if (swapped > 0)
                    return {Label(parent.m->label[k1]),
                            Label(parent.m->label[k2])};
                else
                    return {Label(parent.m->label[k2]),
                            Label(parent.m->label[k1])};
            }

        private:
            model const& parent;
            size_t k1, k2;
            size_t k1_offset, k2_offset, k_comb;
            int swapped;
        };

        size_t nr_labels() const {
            if (traits::is_dynamic_label<Label>::value) {
                return m->nr_class;
            } else {
                return traits::label_traits<Label>::nr_labels;
            }
        }

        size_t nr_classifiers() const {
            return nr_labels() * (nr_labels() - 1) / 2;
        }

        using decision_type =
            std::conditional_t<traits::is_binary_label<Label>::value,
                               double,
                               std::conditional_t<traits::is_dynamic_label<Label>::value,
                                                  std::vector<double>,
                                                  std::array<double, NRC>>>;

        using nSV_type = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                            std::vector<size_t>,
                                            std::array<size_t, NRL>>;
        using label_arr_t = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                               std::vector<label_type>,
                                               std::array<label_type, NRL>>;
        using classifier_arr_t = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                                    std::vector<classifier_type>,
                                                    std::array<classifier_type, NRC>>;

        model () : prob(0), m(nullptr) {}

        model (problem_t && problem, parameters_t const& parameters)
            : prob(std::move(problem)),
              params_(parameters)
        {
            struct svm_problem svm_prob = prob.generate();
            const char * err = svm_check_parameter(&svm_prob, params_.svm_params_ptr());
            if (err) {
                std::string err_str(err);
                throw std::runtime_error(err_str);
            }
            m = svm_train(&svm_prob, params_.svm_params_ptr());
            if (!traits::is_dynamic_label<Label>::value
                && size_t(m->nr_class) != traits::label_traits<Label>::nr_labels)
            {
                throw std::runtime_error("inconsistent number of label values");
            }
            if (std::any_of(m->rho, m->rho + nr_classifiers(),
                            [] (double r) { return std::isnan(r); }))
                throw std::runtime_error("SVM returned NaN. Specified nu is infeasible.");
            init_perm();
        }

        model (model const&) = delete;
        model & operator= (model const&) = delete;

        model (model && other)
            : prob(std::move(other.prob)),
              params_(other.params_),
              m(other.m)
        {
            other.m = nullptr;
            init_perm();
        }

        model & operator= (model && other) {
            prob = std::move(other.prob);
            params_ = std::move(other.params_);
            if (m)
                svm_free_and_destroy_model(&m);
            m = other.m;
            other.m = nullptr;
            init_perm();
            return *this;
        }

        ~model () noexcept {
            if (m)
                svm_free_and_destroy_model(&m); // WTF
        }

        label_arr_t labels() const {
            auto ret = detail::container_factory<label_arr_t>::create(nr_labels());
            for (size_t k = 0; k < ret.size(); ++k) {
                ret[k] = m->label[perm_inv[k]];
            }
            return ret;
        }

        template <typename...,
                  typename L = Label,
                  typename = std::enable_if_t<!traits::is_dynamic_label<L>::value>,
                  typename Indices = std::make_index_sequence<NRC>>
        classifier_arr_t classifiers() const {
            return classifiers_impl(Indices {});
        }

        template <typename...,
                  typename L = Label,
                  typename = std::enable_if_t<traits::is_dynamic_label<L>::value>>
        classifier_arr_t classifiers() const {
            classifier_arr_t cls;
            cls.reserve(nr_classifiers());
            for (size_t r1 = 0; r1 < nr_labels() - 1; ++r1)
                for (size_t r2 = r1 + 1; r2 < nr_labels(); ++r2)
                    cls.emplace_back(*this, perm_inv[r1], perm_inv[r2]);
            return cls;
        }

        classifier_type classifier(Label l1, Label l2) const {
            auto ls = labels();
            size_t r1, r2;
            for (size_t r = 0; r < nr_labels(); ++r) {
                if (l1 == ls[r])
                    r1 = r;
                if (l2 == ls[r])
                    r2 = r;
            }
            return classifier_type {*this, perm_inv[r1], perm_inv[r2]};
        }

        template <typename...,
                  typename L = Label,
                  typename = std::enable_if_t<traits::is_binary_label<L>::value
                                              || traits::is_dynamic_label<L>::value>>
        classifier_type classifier() const {
            return classifier_type {*this, perm_inv[0], perm_inv[1]};
        }

        template <typename Problem = problem_t,
                  typename = std::enable_if_t<!Problem::is_precomputed>>
        std::pair<Label, decision_type> raw_eval(input_container_type const& xj) const {
            std::pair<Label, decision_type> ret {
                Label{},
                detail::container_factory<decision_type>::create(nr_classifiers())
            };
            ret.first = Label{svm_predict_values(m, xj.ptr(),
                                                 detail::container_factory<decision_type>::ptr(ret.second))};
            return ret;
        }

        template <typename Problem = problem_t,
                  typename = std::enable_if_t<Problem::is_precomputed>,
                  bool dummy = false>
        std::pair<Label, decision_type> raw_eval(input_container_type const& xj) const {
            dataset kernelized = prob.kernelize(xj);
            std::pair<Label, decision_type> ret {
                Label{},
                detail::container_factory<decision_type>::create(nr_classifiers())
            };
            ret.first = Label{svm_predict_values(m, kernelized.ptr(),
                                                 detail::container_factory<decision_type>::ptr(ret.second))};
            return ret;
        }

        std::pair<Label, decision_type> operator() (input_container_type const& xj) const {
            auto p = raw_eval(xj);
            permute(p.second);
            return p;
        }

        decision_type rho() const {
            auto r = detail::container_factory<decision_type>::copy(
                m->rho, m->rho + nr_classifiers());
            permute(r);
            return r;
        }

        nSV_type nSV () const {
            return detail::container_factory<nSV_type>::copy(m->nSV,
                                                             m->nSV + nr_labels());
        }

        size_t dim () const {
            return prob.dim();
        }

        parameters_t const& params () const {
            return params_;
        }

        bool empty() const {
            return m == nullptr;
        }

        template <typename Tag, typename Model>
        friend struct serialization::model_serializer;

    private:
        void init_perm () {
            // prep member vars
            perm_inv = detail::container_factory<perm_t>::create(nr_labels());
            permc = detail::container_factory<permc_t>::create(nr_classifiers());
            permc_signs = detail::container_factory<permc_signs_t>::create(nr_classifiers());

            // permutation of label indices
            std::iota(perm_inv.begin(), perm_inv.end(), 0);
            std::sort(perm_inv.begin(), perm_inv.end(),
                      [this] (size_t k1, size_t k2) {
                          return label_type(m->label[k1]) < label_type(m->label[k2]);
                      });

            // invert
            auto perm = detail::container_factory<perm_t>::create(nr_labels());
            for (size_t k = 0; k < perm.size(); ++k)
                perm[perm_inv[k]] = k;

            // permutation of classifier indices
            auto permc_inv = detail::container_factory<permc_t>::create(nr_classifiers());
            auto signs = detail::container_factory<permc_signs_t>::create(nr_classifiers());
            auto it = permc_inv.begin();
            auto sg_it = signs.begin();
            for (size_t k1 = 0; k1 < perm.size() - 1; ++k1) {
                for (size_t k2 = k1 + 1; k2 < perm.size(); ++k2, ++it, ++sg_it) {
                    size_t r1 = perm[k1];
                    size_t r2 = perm[k2];
                    if (r1 > r2) {
                        std::swap(r1, r2);
                        *sg_it = -1;
                    } else
                        *sg_it = 1;
                    *it = (2 * perm.size() - 3 - r1) * r1 / 2 + r2 - 1;
                }
            }
            // invert
            for (size_t k = 0; k < permc.size(); ++k) {
                permc[permc_inv[k]] = k;
                permc_signs[permc_inv[k]] = signs[k];
            }
        }

        template <typename Container>
        void permute(Container & arr) const {
            for (size_t c = 0; c < arr.size(); ++c) {
                if (permc[c] >= arr.size())
                    continue;
                size_t p, h;
                for (h = c; (p = permc[h]) != c; h = p) {
                    std::swap(arr[h], arr[p]);
                    permc[h] += arr.size();
                }
                permc[h] += arr.size();
            }
            for (size_t c = 0; c < arr.size(); ++c) {
                arr[c] *= permc_signs[c];
                permc[c] -= arr.size();
            }
        }

        void permute(double & a) const {
            a *= permc_signs[0];
        }

        template <size_t... R>
        classifier_arr_t classifiers_impl (std::index_sequence<R...>) const {
            auto get_cl = [&](size_t r) -> classifier_type {
                size_t r1 = 0, r2 = r + 1;
                while (r2 >= nr_labels()) {
                    ++r1;
                    r2 -= nr_labels() - 1 - r1;
                }
                return {*this, perm_inv[r1], perm_inv[r2]};
            };
            return {get_cl(R)...};
        }

        problem_t prob;
        parameters_t params_;
        struct svm_model * m;

        using perm_t = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                          std::vector<size_t>,
                                          std::array<size_t, NRL>>;
        using permc_t = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                           std::vector<size_t>,
                                           std::array<size_t, NRC>>;
        using permc_signs_t = std::conditional_t<traits::is_dynamic_label<Label>::value,
                                                 std::vector<int>,
                                                 std::array<int, NRC>>;
        perm_t perm_inv;
        mutable permc_t permc;
        permc_signs_t permc_signs;
    };

}
