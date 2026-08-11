#include "Exciter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    ExciterConfig cfg;
    cfg.dim = 6; // N = 64 — cheap smoke
    cfg.seed = 1;
    cfg.input_scaling = 0.02f;
    cfg.weight_scaling = 0.02f;

    auto exc = Exciter::Create(cfg);
    const size_t n = exc->Size();

    std::vector<float> field(n, 0.0f);
    field[0] = 1.0f;

    const float* y = exc->ExciteCube(field.data());

    float y_min = y[0];
    float y_max = y[0];
    float y_abs_sum = 0.0f;
    bool all_finite = true;

    for (size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(y[i]))
            all_finite = false;
        y_min = std::min(y_min, y[i]);
        y_max = std::max(y_max, y[i]);
        y_abs_sum += std::fabs(y[i]);
    }

    std::cout << "HypercubeEtalon smoke\n"
              << "  dim=" << exc->Dim() << "  N=" << n << '\n'
              << "  y[0]=" << y[0]
              << "  min=" << y_min
              << "  max=" << y_max
              << "  mean_abs=" << (y_abs_sum / static_cast<float>(n)) << '\n';

    if (!all_finite)
    {
        std::cerr << "FAIL: non-finite output\n";
        return EXIT_FAILURE;
    }

    std::cout << "OK\n";
    return EXIT_SUCCESS;
}
