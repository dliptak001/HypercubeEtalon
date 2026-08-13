#include "BaselineExtractor.h"
#include "RamanDataset.h"

#include <stdexcept>

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
    etalon_.CollectBatch(split.spectra, split.baselines);
}

void BaselineExtractor::Train()
{
    etalon_.TrainOnCollected();
}

void BaselineExtractor::Predict(std::span<const float> spectrum,
                                std::span<float> baseline)
{
    if (baseline.size() != etalon_.NumOutputs())
    {
        throw std::invalid_argument(
            "BaselineExtractor::Predict: baseline size must equal N");
    }
    etalon_.Run(spectrum);
    etalon_.readout().PredictRaw(etalon_.LastFeatures().data(), baseline.data());
}
