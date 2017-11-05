#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "serialization_test.hpp"
#include "kernel/linear_precomputed.hpp"


TEST_CASE("serializer-ascii-builtin") {
    serializer_test<svm::kernel::linear, svm::ascii_tag>(25, 2500, 0.98, "ascii-builtin");
}

TEST_CASE("serializer-ascii-precomputed") {
    serializer_test<svm::kernel::linear_precomputed, svm::ascii_tag>(25, 2500, 0.98, "ascii-precomputed");
}
