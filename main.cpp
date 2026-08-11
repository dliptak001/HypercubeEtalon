#include "Etalon.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

// dim=5 → N=32: cheap enough for CI; Readout wants dim >= 5 for pooled stacks.
constexpr size_t kDim = 5;
constexpr size_t kN = 32;

std::vector<float> MakeField(size_t n, int label, int variant)
{
    // Class 0: positive ramp in low half; class 1: negative ramp.
    // `variant` adds a small deterministic tweak so samples are not identical.
    std::vector<float> x(n, 0.0f);
    const float sign = (label == 0) ? 1.0f : -1.0f;
    for (size_t i = 0; i < n / 2; ++i)
        x[i] = sign * (0.2f + 0.8f * static_cast<float>(i) / static_cast<float>(n));
    x[static_cast<size_t>(variant) % n] += 0.02f * static_cast<float>(variant + 1);
    return x;
}

bool AllFinite(std::span<const float> y)
{
    for (float v : y)
    {
        if (!std::isfinite(v))
            return false;
    }
    return true;
}

EtalonConfig MakeCfg()
{
    EtalonConfig cfg;
    cfg.exciter.dim = kDim;
    cfg.exciter.seed = 1;
    // Strong enough that signed-ramp classes stay separable after ExciteCube.
    cfg.exciter.input_scaling = 1.0f;
    cfg.exciter.weight_scaling = 0.5f;

    cfg.readout.dim = 0; // auto = exciter.dim
    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 120;
    cfg.readout.batch_size = 8;
    cfg.readout.num_threads = 1;
    cfg.readout.restore_best_epoch = false;
    cfg.readout.seed = 7;
    cfg.readout.num_layers = 1;
    cfg.readout.lr_max = 0.003f;
    return cfg;
}

} // namespace

int main()
{
    try
    {
        // ----- 1) Contract: sizes, non-mutation, LastFeatures -----
        {
            Etalon et(MakeCfg());
            if (et.N() != kN || et.Dim() != kDim || et.FeatureSize() != kN
                || et.NumOutputs() != 2)
            {
                std::cerr << "FAIL: size contract N=" << et.N()
                          << " Dim=" << et.Dim()
                          << " F=" << et.FeatureSize()
                          << " outs=" << et.NumOutputs() << '\n';
                return EXIT_FAILURE;
            }

            auto x = MakeField(kN, 0, 0);
            const auto x_copy = x;
            et.Run(x);
            if (x != x_copy)
            {
                std::cerr << "FAIL: Run mutated caller field\n";
                return EXIT_FAILURE;
            }
            if (et.LastFeatures().size() != kN || !AllFinite(et.LastFeatures()))
            {
                std::cerr << "FAIL: LastFeatures bad after Run\n";
                return EXIT_FAILURE;
            }

            // Second Run must be deterministic for the same field.
            std::vector<float> feat_a(et.LastFeatures().begin(),
                                      et.LastFeatures().end());
            et.Run(x);
            if (et.LastFeatures().size() != feat_a.size())
            {
                std::cerr << "FAIL: LastFeatures size drift\n";
                return EXIT_FAILURE;
            }
            for (size_t i = 0; i < feat_a.size(); ++i)
            {
                if (std::fabs(et.LastFeatures()[i] - feat_a[i]) > 1e-6f)
                {
                    std::cerr << "FAIL: Run not deterministic\n";
                    return EXIT_FAILURE;
                }
            }

            std::cout << "HypercubeEtalon smoke\n"
                      << "  Etalon dim=" << et.Dim() << "  N=" << et.N() << '\n'
                      << "  y[0]=" << feat_a[0] << '\n';
        }

        // ----- 2) Collect → TrainOnCollected → PredictClass -----
        {
            Etalon et(MakeCfg());
            constexpr int kPerClass = 16;
            constexpr int kClasses = 2;

            for (int lab = 0; lab < kClasses; ++lab)
            {
                for (int v = 0; v < kPerClass; ++v)
                {
                    auto field = MakeField(kN, lab, v);
                    et.Collect(field, lab);
                }
            }

            if (et.NumCollected() != static_cast<size_t>(kPerClass * kClasses))
            {
                std::cerr << "FAIL: NumCollected=" << et.NumCollected() << '\n';
                return EXIT_FAILURE;
            }

            et.TrainOnCollected();
            const double acc = et.AccuracyOnCollected();
            std::cout << "  " << et.ReadoutArchSummary() << '\n'
                      << "  train accuracy=" << acc << '\n';

            if (!(acc >= 0.75))
            {
                std::cerr << "FAIL: train accuracy too low (" << acc << ")\n";
                return EXIT_FAILURE;
            }

            // Inference path (no collect noise; fresh map each call).
            for (int lab = 0; lab < kClasses; ++lab)
            {
                auto field = MakeField(kN, lab, lab); // in-distribution variant
                const auto logits = et.Predict(field);
                if (!AllFinite(logits))
                {
                    std::cerr << "FAIL: non-finite logits for label " << lab
                              << '\n';
                    return EXIT_FAILURE;
                }
                const int pred = et.PredictClass(field);
                if (pred != lab)
                {
                    std::cerr << "FAIL: PredictClass got " << pred
                              << " expected " << lab << '\n';
                    return EXIT_FAILURE;
                }
            }

            // Weight round-trip.
            auto w = et.GetReadoutWeights();
            if (w.empty())
            {
                std::cerr << "FAIL: empty readout weights\n";
                return EXIT_FAILURE;
            }
            et.SetReadoutWeights(w);
            const double acc2 = et.AccuracyOnCollected();
            if (std::fabs(acc2 - acc) > 1e-5)
            {
                std::cerr << "FAIL: weight round-trip accuracy drift "
                          << acc << " -> " << acc2 << '\n';
                return EXIT_FAILURE;
            }
        }

        // ----- 3) Bypass: features == field -----
        {
            EtalonConfig cfg = MakeCfg();
            cfg.bypass_exciter = true;
            Etalon et(cfg);
            auto x = MakeField(kN, 1, 3);
            et.Run(x);
            for (size_t i = 0; i < kN; ++i)
            {
                if (std::fabs(et.LastFeatures()[i] - x[i]) > 1e-7f)
                {
                    std::cerr << "FAIL: bypass_exciter did not copy field\n";
                    return EXIT_FAILURE;
                }
            }
        }

        // ----- 4) Bulk collect path -----
        {
            Etalon et(MakeCfg());
            constexpr size_t kCount = 8;
            std::vector<float> flat(kCount * kN);
            std::vector<int> labels(kCount);
            for (size_t i = 0; i < kCount; ++i)
            {
                const int lab = static_cast<int>(i % 2);
                labels[i] = lab;
                auto f = MakeField(kN, lab, static_cast<int>(i));
                std::copy(f.begin(), f.end(),
                          flat.begin() + static_cast<std::ptrdiff_t>(i * kN));
            }
            et.CollectBatch(flat, labels);
            if (et.NumCollected() != kCount)
            {
                std::cerr << "FAIL: CollectBatch count\n";
                return EXIT_FAILURE;
            }
            et.ClearCollected();
            if (et.NumCollected() != 0)
            {
                std::cerr << "FAIL: ClearCollected\n";
                return EXIT_FAILURE;
            }
        }

        std::cout << "OK\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "FAIL: exception: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
