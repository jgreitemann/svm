#include "dataset.hpp"


using namespace svm;


// ----- data_view::const_iterator -----

data_view::const_iterator::const_iterator (struct svm_node const * ptr,
                                           int index)
    : ptr(ptr), index(index) {}

data_view::const_iterator & data_view::const_iterator::operator++ () {
    if (ptr->index == index)
        ++ptr;
    ++index;
    return *this;
}

data_view::const_iterator data_view::const_iterator::operator++ (int) {
    const_iterator old(*this);
    ++(*this);
    return old;
}

double data_view::const_iterator::operator* () const {
    if (ptr->index == index)
        return ptr->value;
    return 0.;
}

bool data_view::const_iterator::is_end () const {
    return index == -1 || ptr->index == -1;
}

bool svm::operator== (data_view::const_iterator lhs,
                      data_view::const_iterator rhs)
{
    if (lhs.is_end() && rhs.is_end())
        return true;
    return lhs.index == rhs.index && lhs.ptr == rhs.ptr;
}

bool svm::operator!= (data_view::const_iterator lhs,
                      data_view::const_iterator rhs)
{
    return !(lhs == rhs);
}


// ----- data_view -----


data_view::data_view ()
    : begin_ptr(nullptr), start_index(-1) {}

data_view::data_view (struct svm_node const * ptr, int start_index)
    : begin_ptr(ptr), start_index(start_index) {}

data_view::data_view (dataset const& ds) {
    *this = ds.view();
}

data_view::const_iterator data_view::begin () const {
    if (start_index == -1)
        return end();
    return const_iterator(begin_ptr);
}
data_view::const_iterator data_view::end () const {
    return const_iterator(nullptr, -1);
}

double data_view::dot (data_view other) const {
    data_view::const_iterator lhs = begin();
    data_view::const_iterator rhs = other.begin();
    double sum = 0;
    for (; lhs != end() && rhs != other.end(); ++lhs, ++rhs) {
        sum += *lhs * *rhs;
    }
    return sum;
}

double svm::dot (data_view lhs, data_view rhs) {
    return lhs.dot(rhs);
}
