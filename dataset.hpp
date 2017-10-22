#pragma once

#include "svm.h"

#include <vector>


namespace svm {

    class dataset {
    public:
        typedef typename std::vector<svm_node>::iterator iterator;
        typedef typename std::vector<svm_node>::const_iterator const_iterator;

        template <typename OutputIterator>
        dataset (OutputIterator begin, OutputIterator end) {
            nodify(begin, end);
        }

        template <typename Container>
        dataset (Container const& c) {
            nodify(c.begin(), c.end());
        }

        double dot (dataset const& other) const {
            auto lhs = begin();
            auto rhs = other.begin();
            double sum = 0;
            while (lhs != end() && rhs != other.end()) {
                if (lhs->index == rhs->index) {
                    sum += lhs->value * rhs->value;
                    ++lhs;
                    ++rhs;
                } else if (lhs->index < rhs->index) {
                    ++lhs;
                } else {
                    ++rhs;
                }
            }
            return sum;
        }

        iterator begin() { return data.begin(); }
        const_iterator begin() const { return data.begin(); }
        iterator end() { return --data.end(); }
        const_iterator end() const { return --data.end(); }

        struct svm_node * ptr () {
            return data.data();
        }

        struct svm_node const * ptr () const {
            return data.data();
        }

    private:
        template <typename OutputIterator>
        void nodify (OutputIterator begin, OutputIterator end) {
            for (int i = 0; begin != end; ++i, ++begin)
                if (*begin != 0)
                    data.push_back({ .index = i, .value = *begin });
            data.push_back({ .index = -1 });
        }

        std::vector<struct svm_node> data;
    };

}
