#pragma once

#include "Etalon.h"

#include <span>
#include <string>

struct RamanSplit;

constexpr size_t kDim = 11;
constexpr size_t kN = size_t{1} << kDim;

inline EtalonConfig MakeConfig()
{
    EtalonConfig cfg;
    cfg.exciter.dim = kDim;
    cfg.exciter.seed = 3458567978345987ull;
    cfg.exciter.subcube_dim = 5;
    cfg.exciter.input_scaling = 1.0;
    cfg.exciter.weight_scaling = 0.15;

    cfg.readout.epochs = 100;
    cfg.readout.dim = 0; // auto
    cfg.readout.num_outputs = static_cast<int>(kN);
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.batch_size = 48;
    cfg.readout.conv_channels = 1;
    cfg.readout.channel_growth = 1;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = false;
    cfg.readout.lr_max = 0.003f;
    cfg.readout.lr_min_frac = 0.04;
    cfg.readout.restore_best_epoch = true;

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

    void SaveReadout(const std::string& path_stem) const;
    void LoadReadout(const std::string& path_stem);

private:
    Etalon etalon_;
};
