#pragma once


namespace svm {

    template <typename Tag, typename Model>
    struct model_serializer;

    template <typename Tag, typename Problem>
    struct problem_serializer;

}

#include "ascii_serialization.hpp"
