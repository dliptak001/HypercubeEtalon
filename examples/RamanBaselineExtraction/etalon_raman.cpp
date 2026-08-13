#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanScore.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

static constexpr int kTrainSamples = 100;
static constexpr int kTestSamples = 50;

static constexpr const char* kDataRoot = "C:/MLPlayground/Datasets/data";

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

        BaselineExtractor ex;
        if (ex.N() != kRamanBins)
        {
            throw std::logic_error(
                "extractor N does not match dataset bin count");
        }

        std::printf("etalon_raman: train=%zu val=%zu N=%zu\n",
                    train.count, test.count, ex.N());
        std::fflush(stdout);

        ex.Collect(train);
        ex.Train();

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
