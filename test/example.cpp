#include <svm-wrapper.hpp>
#include <array>
#include <random>

int main () {
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform(-1, 1);

    using kernel_t = svm::kernel::polynomial<2>;
    using data_t = std::array<double, 2>;
    svm::problem<kernel_t> prob(2);
    for (size_t i = 0; i < 100; ++i) {
        data_t x {uniform(rng), uniform(rng)};
        int y = (x[0]*x[0]+x[1]*x[1] < 0.5) ? 1 : -1;
        prob.add_sample(x, y);
    }

    svm::model<kernel_t> model(std::move(prob), svm::parameters<kernel_t>(0.4));
}
