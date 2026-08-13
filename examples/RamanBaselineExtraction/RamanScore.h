#pragma once

#include "BaselineExtractor.h"
#include "RamanDataset.h"

#include <cmath>
#include <stdexcept>
#include <vector>

inline double RamanRmse(BaselineExtractor& ex, const RamanSplit& split)
{
    if (split.count == 0)
        throw std::invalid_argument("RamanRmse: empty split");

    std::vector<float> pred(ex.N());
    double sum_mse = 0.0;
    for (size_t i = 0; i < split.count; ++i)
    {
        ex.Predict(split.Spectrum(i), pred);
        const auto label = split.Baseline(i);
        double acc = 0.0;
        for (size_t j = 0; j < pred.size(); ++j)
        {
            const double e = static_cast<double>(label[j])
                             - static_cast<double>(pred[j]);
            acc += e * e;
        }
        sum_mse += acc / static_cast<double>(pred.size());
    }
    return std::sqrt(sum_mse / static_cast<double>(split.count));
}
