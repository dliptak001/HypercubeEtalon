#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanScore.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

static constexpr int kTrainSamples = 1000;
static constexpr int kTestSamples = 100;
static constexpr bool kRunBypass = true;

static constexpr const char* kDataRoot = "C:/HypercubeEtalon/RamanSpectra";

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

        EtalonConfig cfg = MakeConfig();
        cfg.bypass_exciter = kRunBypass;
        cfg.readout.epoch_tick = EpochTick;

        BaselineExtractor ex(cfg);
        if (ex.N() != kN)
            throw std::logic_error("extractor N must equal kN");
        s_ex = &ex;
        s_train = &train;

        std::printf("etalon_raman: train=%zu val=%zu N=%zu bypass=%s\n",
                    train.count, test.count, ex.N(),
                    kRunBypass ? "true" : "false");
        std::fflush(stdout);

        ex.Collect(train);
        std::printf("etalon_raman: collected\n");
        std::fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        ex.Train();
        const auto t1 = std::chrono::steady_clock::now();
        const double train_secs =
            std::chrono::duration<double>(t1 - t0).count();
        std::printf("etalon_raman: trained time=%.2fs\n", train_secs);
        std::fflush(stdout);

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
