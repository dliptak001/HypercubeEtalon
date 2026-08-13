#pragma once

#include "Etalon.h"

#include <span>

struct RamanSplit;

constexpr size_t kDim = 11;
constexpr size_t kN = size_t{1} << kDim;

inline EtalonConfig MakeConfig()
{
    EtalonConfig cfg;
    cfg.exciter.dim = kDim;
    cfg.exciter.halvings = 7;
    cfg.readout.dim = 0;
    cfg.readout.num_outputs = static_cast<int>(kN);
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.epochs = 20;
    cfg.readout.activation = ReadoutActivation::NONE;

    cfg.collect_threads = 1;

    return cfg;
}

class BaselineExtractor
{
public:
    BaselineExtractor();
    explicit BaselineExtractor(const EtalonConfig& cfg);
    ~BaselineExtractor() = default;

    BaselineExtractor(const BaselineExtractor&) = delete;
    BaselineExtractor& operator=(const BaselineExtractor&) = delete;
    BaselineExtractor(BaselineExtractor&&) noexcept = default;
    BaselineExtractor& operator=(BaselineExtractor&&) noexcept = default;

    [[nodiscard]] size_t Dim() const { return etalon_.Dim(); }
    [[nodiscard]] size_t N() const { return etalon_.N(); }
    [[nodiscard]] size_t NumOutputs() const { return etalon_.NumOutputs(); }

    [[nodiscard]] const EtalonConfig& config() const { return etalon_.config(); }
    [[nodiscard]] Etalon& etalon() { return etalon_; }
    [[nodiscard]] const Etalon& etalon() const { return etalon_; }

    void Collect(const RamanSplit& split);
    void Train();
    void Predict(std::span<const float> spectrum, std::span<float> baseline);

private:
    Etalon etalon_;
};
