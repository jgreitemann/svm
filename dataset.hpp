#pragma once

#include "svm.h"

#include <vector>


namespace svm {

    class dataset {
    public:
        template <typename OutputIterator>
        dataset (OutputIterator begin, OutputIterator end) {
            nodify(begin, end);
        }

        template <typename Container>
        dataset (Container const& c) {
            nodify(c.begin(), c.end());
        }

        struct svm_node * ptr () {
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
