#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

static constexpr size_t kRamanBins = 2048;

struct RamanSplit
{
    size_t count = 0;
    std::vector<float> spectra;
    std::vector<float> baselines;

    [[nodiscard]] std::span<const float> Spectrum(size_t i) const
    {
        return {spectra.data() + i * kRamanBins, kRamanBins};
    }

    [[nodiscard]] std::span<const float> Baseline(size_t i) const
    {
        return {baselines.data() + i * kRamanBins, kRamanBins};
    }
};

RamanSplit LoadRamanSplit(const std::filesystem::path& dir, size_t prefix);
