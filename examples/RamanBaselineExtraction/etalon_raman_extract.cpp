#include "RamanDataset.h"
#include "RamanExtract.h"

#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr bool kRunBypass = false;
static constexpr const char* kDataRoot = "C:/HypercubeEtalon/RamanSpectraLCO";
static constexpr const char* kSplit = "Training";//"Training";
static constexpr const char* kModelStem = "C:/HypercubeEtalon/RamanModels/readout_exciter";
// Dataset file numbers: Training/0.data.txt, Training/42.data.txt, …
static constexpr int kIndices[] = {3811,3812,3813,3814};
static constexpr const char* kOutDir = "C:/HypercubeEtalon/RamanModels/extracted";

int main()
{
    int exit_code = 1;
    try
    {
        const std::span<const int> indices(kIndices);
        const auto split_dir =
            std::filesystem::path(kDataRoot) / kSplit;
        const std::string stem(kModelStem);
        const std::filesystem::path out_dir(kOutDir);

        auto ex = LoadExtractor(stem, kRunBypass);
        const auto split = LoadRamanIndices(split_dir, indices);

        std::vector<float> preds(split.count * kN);
        ExtractSplit(ex, split, preds);
        WritePredictions(out_dir, indices, preds, stem, split_dir);

        std::printf("etalon_raman_extract: extracted n=%zu stem=%s -> %s\n",
                    split.count, stem.c_str(), out_dir.string().c_str());
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "etalon_raman_extract: %s\n", e.what());
    }
    return exit_code;
}
