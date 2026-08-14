#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanScore.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int kTrainSamples = 10000;
static constexpr int kTestSamples = 1000;
static constexpr bool kRunBypass = false;
static constexpr bool kSkipTrain = false;

static constexpr const char* kDataRoot = "C:/HypercubeEtalon/RamanSpectra";
static constexpr const char* kModelStem = "C:/HypercubeEtalon/RamanModels/readout";

static BaselineExtractor* s_ex = nullptr;
static const RamanSplit* s_train = nullptr;

static void EpochTick(int epoch, double)
{
    const double train_rmse = RamanRmseOnCollected(*s_ex, *s_train);
    std::printf("etalon_raman: epoch=%d train_rmse=%.6f\n", epoch, train_rmse);
    std::fflush(stdout);
}

int main()
{
    int exit_code = 1;
    try
    {
        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "kTrainSamples / kTestSamples must be >= 0 (0 = whole split)");
        }

        const std::filesystem::path root(kDataRoot);
        const auto train = LoadRamanSplit(
            root / "Training", static_cast<size_t>(kTrainSamples));
        const auto test = LoadRamanSplit(
            root / "Validation", static_cast<size_t>(kTestSamples));

        if (train.count == 0)
            throw std::runtime_error("empty training split");

        const std::string stem =
            std::string(kModelStem) + (kRunBypass ? "_bypass" : "_exciter");

        EtalonConfig cfg = MakeConfig();
        cfg.bypass_exciter = kRunBypass;
        if (!kSkipTrain)
            cfg.readout.epoch_tick = EpochTick;

        BaselineExtractor ex(cfg);
        if (ex.N() != kN)
            throw std::logic_error("extractor N must equal kN");
        s_ex = &ex;
        s_train = &train;

        std::printf("etalon_raman: train=%zu val=%zu N=%zu bypass=%s skip_train=%s\n",
                    train.count, test.count, ex.N(),
                    kRunBypass ? "true" : "false",
                    kSkipTrain ? "true" : "false");
        std::fflush(stdout);

        if (kSkipTrain)
        {
            ex.LoadReadout(stem);
            if (!ex.etalon().readout().IsTrained())
            {
                throw std::runtime_error(
                    "etalon_raman: LoadReadout did not mark trained");
            }
            std::printf("etalon_raman: loaded %s.hcnw\n", stem.c_str());
            std::fflush(stdout);
        }
        else
        {
            ex.Collect(train);
            std::printf("etalon_raman: collected\n");
            std::fflush(stdout);

            const auto t0 = std::chrono::steady_clock::now();
            ex.Train();
            const auto t1 = std::chrono::steady_clock::now();
            const double train_secs =
                std::chrono::duration<double>(t1 - t0).count();
            std::printf("etalon_raman: trained time=%.2fs best_epoch=%d\n",
                        train_secs, ex.etalon().readout().BestEpoch());
            std::fflush(stdout);

            ex.SaveReadout(stem);
            {
                BaselineExtractor check(cfg);
                check.LoadReadout(stem);
                std::vector<float> a(kN), b(kN);
                const auto probe = train.Spectrum(0);
                ex.Predict(probe, a);
                check.Predict(probe, b);
                for (size_t i = 0; i < kN; ++i)
                {
                    if (std::fabs(a[i] - b[i]) > 1e-3f)
                    {
                        throw std::runtime_error(
                            "etalon_raman: saved readout failed reload check");
                    }
                }
            }
            std::printf("etalon_raman: saved %s.hcnw %s.arch.json\n",
                        stem.c_str(), stem.c_str());
            std::fflush(stdout);
        }

        const double train_rmse = RamanRmse(ex, train);
        const double val_rmse = RamanRmse(ex, test);
        std::printf("etalon_raman: train_rmse=%.6f val_rmse=%.6f\n",
                    train_rmse, val_rmse);
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "etalon_raman: %s\n", e.what());
    }
    return exit_code;
}
