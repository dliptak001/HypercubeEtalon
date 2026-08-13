#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanScore.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

static constexpr int kTrainSamples = 1000;
static constexpr int kTestSamples = 100;
static constexpr bool kRunBypass = false;

static constexpr const char* kDataRoot = "C:/MLPlayground/Datasets/data";

static int s_epoch_on_line = 0;

static void EpochTick()
{
    std::fputc('.', stdout);
    if (++s_epoch_on_line == 10)
    {
        std::fputc('\n', stdout);
        s_epoch_on_line = 0;
    }
    std::fflush(stdout);
}

static void FinishEpochTicks()
{
    if (s_epoch_on_line == 0)
        return;
    std::fputc('\n', stdout);
    std::fflush(stdout);
    s_epoch_on_line = 0;
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

        std::printf("etalon_raman: train=%zu val=%zu N=%zu bypass=%s\n",
                    train.count, test.count, ex.N(),
                    kRunBypass ? "true" : "false");
        std::fflush(stdout);

        ex.Collect(train);
        std::printf("etalon_raman: collected\n");
        std::fflush(stdout);

        ex.Train();
        FinishEpochTicks();
        std::printf("etalon_raman: trained\n");
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
