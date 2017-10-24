#pragma once

#include "svm.h"

#include <vector>


namespace svm {

    // forward declaration
    class dataset;

    class data_view {
    public:
        class const_iterator {
        public:
            const_iterator (struct svm_node const * ptr, int index = 1);
            const_iterator & operator++ ();
            const_iterator operator++ (int);
            double operator* () const;
            bool is_end () const;
            friend bool operator== (const_iterator, const_iterator);
            friend bool operator!= (const_iterator, const_iterator);
        private:
            struct svm_node const * ptr;
            int index;
        };

        data_view ();
        data_view (struct svm_node const * ptr, int start_index = 1);
        data_view (dataset const&);

        const_iterator begin () const;
        const_iterator end () const;

        double dot (data_view other) const;
    private:
        struct svm_node const * begin_ptr;
        int start_index;
    };

    bool operator== (data_view::const_iterator, data_view::const_iterator);
    bool operator!= (data_view::const_iterator, data_view::const_iterator);
    double dot (data_view, data_view);

    class dataset {
    public:
        template <typename OutputIterator>
        dataset (OutputIterator begin, OutputIterator end,
                 int start_index = 1, bool skip_zeros = true)
            : start_index(start_index)
        {
            nodify(begin, end, skip_zeros);
        }

        template <typename Container>
        dataset (Container const& c,
                 int start_index = 1, bool skip_zeros = true)
            : start_index(start_index)
        {
            nodify(c.begin(), c.end(), skip_zeros);
        }

        struct svm_node * ptr () {
            return data.data();
        }

        struct svm_node const * ptr () const {
            return data.data();
        }

        data_view view () const {
            return data_view(ptr(), start_index);
        }

    private:
        template <typename OutputIterator>
        void nodify (OutputIterator begin, OutputIterator end, bool skip_zeros) {
            for (int i = start_index; begin != end; ++i, ++begin)
                if (!skip_zeros || *begin != 0)
                    data.push_back({ .index = i, .value = *begin });
            data.push_back({ .index = -1 });
        }

        std::vector<struct svm_node> data;
        int start_index;
    };

}
