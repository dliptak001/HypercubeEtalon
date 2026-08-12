#include "Etalon.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// dim=5 → N=32: cheap enough for CI; roomy enough for a 1-layer pooled stack.
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
        // ----- 0) Dim-range contracts (exceptions, not debug asserts) -----
        {
            ReadoutConfig bad;
            bad.dim = 0;
            bad.num_outputs = 2;
            bad.task = ReadoutTask::Classification;
            bool threw = false;
            try { Readout ro(bad); } catch (const std::invalid_argument&) { threw = true; }
            if (!threw)
            {
                std::cerr << "FAIL: Readout dim=0 should throw\n";
                return EXIT_FAILURE;
            }

            ReadoutConfig deep;
            deep.dim = 4;
            deep.num_layers = 3; // > dim-2
            deep.use_pooling = true;
            deep.num_outputs = 2;
            deep.task = ReadoutTask::Classification;
            threw = false;
            try { Readout ro(deep); } catch (const std::invalid_argument&) { threw = true; }
            if (!threw)
            {
                std::cerr << "FAIL: pooled Readout layers > dim-2 should throw\n";
                return EXIT_FAILURE;
            }

            EtalonConfig d4 = MakeCfg();
            d4.exciter.dim = 4;
            d4.readout.num_layers = 1;
            Etalon et4(d4);
            if (et4.Dim() != 4 || et4.N() != 16)
            {
                std::cerr << "FAIL: dim=4 Etalon sizes\n";
                return EXIT_FAILURE;
            }
            auto x4 = MakeField(16, 0, 0);
            et4.Run(x4);
            if (et4.LastFeatures().size() != 16 || !AllFinite(et4.LastFeatures()))
            {
                std::cerr << "FAIL: dim=4 Run\n";
                return EXIT_FAILURE;
            }
        }

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
            std::vector<float> train_flat(
                static_cast<size_t>(kPerClass * kClasses) * kN);
            std::vector<int> train_labs(
                static_cast<size_t>(kPerClass * kClasses));

            for (int lab = 0; lab < kClasses; ++lab)
            {
                for (int v = 0; v < kPerClass; ++v)
                {
                    const size_t i =
                        static_cast<size_t>(lab * kPerClass + v);
                    auto field = MakeField(kN, lab, v);
                    std::copy(field.begin(), field.end(),
                              train_flat.begin()
                                  + static_cast<std::ptrdiff_t>(i * kN));
                    train_labs[i] = lab;
                    et.Collect(field, lab);
                }
            }

            if (et.NumCollected() != static_cast<size_t>(kPerClass * kClasses))
            {
                std::cerr << "FAIL: NumCollected=" << et.NumCollected() << '\n';
                return EXIT_FAILURE;
            }
            if (et.IsReadoutTrained())
            {
                std::cerr << "FAIL: IsReadoutTrained true before Train\n";
                return EXIT_FAILURE;
            }

            et.TrainOnCollected();
            if (!et.IsReadoutTrained())
            {
                std::cerr << "FAIL: IsReadoutTrained false after Train\n";
                return EXIT_FAILURE;
            }
            const double acc = et.AccuracyOnCollected();
            std::cout << "  " << et.ReadoutArchSummary() << '\n'
                      << "  train accuracy=" << acc << '\n';

            if (!(acc >= 0.75))
            {
                std::cerr << "FAIL: train accuracy too low (" << acc << ")\n";
                return EXIT_FAILURE;
            }

            // Fresh-map Accuracy on the same fields (no collect noise) must
            // match AccuracyOnCollected.
            const double acc_mapped = et.Accuracy(train_flat, train_labs);
            if (std::fabs(acc_mapped - acc) > 1e-5)
            {
                std::cerr << "FAIL: Accuracy() vs AccuracyOnCollected "
                          << acc_mapped << " vs " << acc << '\n';
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

            // HCNW + arch sidecar: Etalon format token, load marks trained.
            const auto stem = (std::filesystem::temp_directory_path()
                               / "etalon_smoke_readout").string();
            et.SaveReadoutHcnnModel(stem);
            {
                std::ifstream arch(stem + ".arch.json");
                const std::string text(
                    (std::istreambuf_iterator<char>(arch)),
                    std::istreambuf_iterator<char>());
                if (text.find(Readout::kArchSidecarFormat) == std::string::npos
                    || text.find("hypercube_esn_readout_arch")
                           != std::string::npos)
                {
                    std::cerr << "FAIL: sidecar format token\n";
                    return EXIT_FAILURE;
                }
            }
            {
                Etalon loaded(MakeCfg());
                if (loaded.IsReadoutTrained())
                {
                    std::cerr << "FAIL: fresh Etalon marked trained\n";
                    return EXIT_FAILURE;
                }
                loaded.LoadReadoutHcnnModel(stem);
                if (!loaded.IsReadoutTrained())
                {
                    std::cerr << "FAIL: LoadReadoutHcnnModel did not mark trained\n";
                    return EXIT_FAILURE;
                }
            }
            std::filesystem::remove(stem + ".arch.json");
            std::filesystem::remove(stem + ".hcnw");
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
