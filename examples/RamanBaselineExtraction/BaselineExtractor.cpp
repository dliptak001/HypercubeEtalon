#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"

#include <cmath>
#include <stdexcept>
#include <vector>

BaselineExtractor::BaselineExtractor()
    : BaselineExtractor(MakeConfig())
{
}

BaselineExtractor::BaselineExtractor(const EtalonConfig& cfg)
    : etalon_(cfg)
{
}

void BaselineExtractor::Collect(const RamanSplit& split)
{
    if (split.spectra.size() != split.count * kN
        || split.baselines.size() != split.count * kN)
    {
        throw std::invalid_argument(
            "BaselineExtractor::Collect: count does not match buffer size");
    }

    RamanNorm::Check();

    std::vector<float> spectra(split.spectra.size());
    std::vector<float> baselines(split.baselines.size());
    std::vector<float> back(kN);
    for (size_t i = 0; i < split.count; ++i)
    {
        const auto spec = split.Spectrum(i);
        const auto lab = split.Baseline(i);
        const auto nrm = RamanNorm::FromSpectrum(spec);
        auto mapped = std::span<float>{spectra.data() + i * kN, kN};
        nrm.Apply(spec, mapped);
        nrm.Apply(lab, {baselines.data() + i * kN, kN});
        nrm.Invert(mapped, back);
        for (size_t j = 0; j < kN; ++j)
        {
            if (std::fabs(back[j] - spec[j]) > 1e-2f)
            {
                throw std::runtime_error(
                    "BaselineExtractor::Collect: spectrum denorm round-trip failed");
            }
        }
    }
    etalon_.CollectBatch(spectra, baselines);
}

void BaselineExtractor::Train()
{
    etalon_.TrainOnCollected();
}

void BaselineExtractor::Predict(std::span<const float> spectrum,
                                std::span<float> baseline)
{
    if (spectrum.size() != etalon_.N()
        || baseline.size() != etalon_.NumOutputs())
    {
        throw std::invalid_argument(
            "BaselineExtractor::Predict: spectrum and baseline size must equal N");
    }

    const auto nrm = RamanNorm::FromSpectrum(spectrum);
    std::vector<float> xn(etalon_.N());
    std::vector<float> yn(etalon_.NumOutputs());
    nrm.Apply(spectrum, xn);
    etalon_.Run(xn);
    etalon_.readout().PredictRaw(etalon_.LastFeatures().data(), yn.data());
    nrm.Invert(yn, baseline);
}
