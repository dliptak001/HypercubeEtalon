/// @file etalon_mnist.cpp
/// @brief MNIST → pack length-N field → Etalon → held-out test.
///
/// Pipeline: IDX load → file-order prefix → PadLowCenter pack → CollectBatch
/// → TrainOnCollected → Accuracy on packed test fields.
///
/// Product path: pack → Exciter → readout → held-out Accuracy.
/// Bypass (pack → readout, no walk) is opt-in via kRunBypass.
/// Optional test-only AWGN on the packed field (train clean). Demo knobs
/// kTestNoiseSweep / Start / End / Step. Off = clean test. On = train once
/// per path, then score start, start+step, ... while <= end and print a
/// table. kRunBypass scores the same grid on the skip-the-walk path too.
///
/// Data: C:\HypercubeEtalon\data (see examples/README.md appendix).
///
/// Not ported from wtf_mnist: geometric aug or the noise-study write-ups.

#include "Etalon.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "pack_field.h"
#include "print_config.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Pack
// =============================================================================

enum class PackMode
{
    PadLow,
    PadLowCenter, // default: 28x28 + centered crop fills N=1024
};

static hcnn::HCNNSpatialEmbedMode ToEmbedMode(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow: return hcnn::HCNNSpatialEmbedMode::PadLow;
    case PackMode::PadLowCenter: return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
    }
    return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
}

static const char* PackModeName(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow: return "PadLow";
    case PackMode::PadLowCenter: return "PadLowCenter";
    }
    return "?";
}

// =============================================================================
// Product knobs
// =============================================================================

// Path-specific readout length. Bypass sees the packed field and converges
// sooner; the Exciter bank wants the longer schedule.
static constexpr int kExciterEpochs = 100;//100;
static constexpr int kBypassEpochs = 20;

static EtalonConfig MakeBaseConfig()
{
    EtalonConfig cfg;

    // PadLow / PadLowCenter need N >= 784 → dim >= 10.
    cfg.exciter.dim = 10;
    cfg.exciter.seed = 38715376369942979ull;
    cfg.exciter.subcube_dim = 4;
    cfg.exciter.input_scaling = 0.25;
    cfg.exciter.weight_scaling = 0.2;

    cfg.readout.dim = 0;
    cfg.readout.num_outputs = 10;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = true;
    cfg.readout.conv_channels = 16;
    cfg.readout.channel_growth = 1;
    cfg.readout.activation = ReadoutActivation::NONE;
    cfg.readout.optimizer = ReadoutOptimizer::Adam;
    cfg.readout.epochs = kExciterEpochs;
    cfg.readout.batch_size = 64;
    cfg.readout.lr_max = 0.002f;
    cfg.readout.lr_min_frac = 0.005;
    cfg.readout.num_threads = 0; // HCNN auto
    cfg.readout.restore_best_epoch = true;
    cfg.readout.best_epoch_holdout_frac = 0.1f;
    cfg.readout.seed = 42;

    return cfg;
}

// =============================================================================
// Demo / task (not product config)
// =============================================================================

static constexpr PackMode kPack = PackMode::PadLowCenter;

// MNIST train is 60000, test 10000. A short demo is 1000 / 500.
static constexpr int kTrainSamples = 60000;
static constexpr int kTestSamples = 10000;
static constexpr float kPad = -1.0f;
static constexpr int kImgSide = 28;
static constexpr int kImgPixels = kImgSide * kImgSide;
// Soft floor on the product path (not chance on 10 classes).
static constexpr double kMinExciterTestAcc = 0.50;

// Occasional consistency check: pack → readout with no Exciter walk.
static constexpr bool kRunBypass = false;

static constexpr bool kTestNoiseSweep = true;
static constexpr float kTestNoiseStart = 0.0f;
static constexpr float kTestNoiseEnd = 1.0f;
static constexpr float kTestNoiseStep = 0.1f;
static constexpr unsigned kTestNoiseSeedBase = 0x7E57u;

// =============================================================================

static void PackSet(const etalon_ex::MnistSet& ds,
                    const hcnn::HCNNSpatialEmbedder& emb,
                    std::vector<float>& fields,
                    std::vector<int>& labels)
{
    const size_t n = static_cast<size_t>(emb.capacity());
    fields.assign(ds.size() * n, 0.0f);
    labels.assign(ds.size(), 0);
    for (size_t i = 0; i < ds.size(); ++i)
    {
        labels[i] = ds.samples[i].label;
        auto row = std::span<float>(fields.data() + i * n, n);
        etalon_ex::PackMnist28(ds.samples[i].pixels.data(), emb, row);
    }
}

static std::vector<float> MakeTestNoiseGrid()
{
    std::vector<float> grid;
    if (!kTestNoiseSweep)
        return grid;
    if (!(kTestNoiseStart >= 0.0f) || !std::isfinite(kTestNoiseStart)
        || !(kTestNoiseEnd >= 0.0f) || !std::isfinite(kTestNoiseEnd)
        || !std::isfinite(kTestNoiseStep) || !(kTestNoiseStep > 0.0f))
    {
        throw std::invalid_argument(
            "etalon_mnist: test noise sweep needs finite start/end >= 0 "
            "and step > 0");
    }
    if (kTestNoiseStart > kTestNoiseEnd)
    {
        throw std::invalid_argument(
            "etalon_mnist: kTestNoiseStart must be <= kTestNoiseEnd");
    }
    if (kTestNoiseEnd <= kTestNoiseStart)
    {
        grid.push_back(kTestNoiseStart);
        return grid;
    }
    const double n = std::floor(
        (static_cast<double>(kTestNoiseEnd) - kTestNoiseStart)
        / kTestNoiseStep + 1e-6) + 1.0;
    if (n > 1000.0)
    {
        throw std::invalid_argument(
            "etalon_mnist: test noise grid would exceed 1000 points");
    }
    const int count = static_cast<int>(n);
    grid.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        grid.push_back(kTestNoiseStart
                       + kTestNoiseStep * static_cast<float>(i));
    }
    return grid;
}

static void PrintTestNoiseReport(const std::vector<float>& grid)
{
    if (!kTestNoiseSweep)
    {
        std::printf("etalon_mnist: test_noise=off\n");
        return;
    }
    std::printf(
        "etalon_mnist: test_noise=sweep start=%g end=%g step=%g "
        "(%zu points) seed_base=0x%X (train once, then each sigma)\n",
        static_cast<double>(kTestNoiseStart),
        static_cast<double>(kTestNoiseEnd),
        static_cast<double>(kTestNoiseStep),
        grid.size(), kTestNoiseSeedBase);
}

/// In-place i.i.d. Gaussian on a packed field (no clamp). No-op if σ <= 0.
static void AddTestFieldNoise(std::span<float> field, size_t sample_index,
                              float sigma)
{
    if (sigma <= 0.0f)
        return;
    std::mt19937 rng(kTestNoiseSeedBase
        + static_cast<unsigned>(sample_index) * 9973u);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (float& v : field)
        v += dist(rng);
}

static double ScoreNoisyTest(Etalon& et,
                             std::span<const float> clean_fields,
                             std::span<const int> labels,
                             float sigma)
{
    const size_t n = et.N();
    if (sigma <= 0.0f)
        return et.Accuracy(clean_fields, labels);

    std::vector<float> noisy(clean_fields.begin(), clean_fields.end());
    if (noisy.size() != labels.size() * n)
        throw std::logic_error("test pack size mismatch");
    for (size_t i = 0; i < labels.size(); ++i)
    {
        AddTestFieldNoise(
            std::span<float>(noisy.data() + i * n, n), i, sigma);
    }
    return et.Accuracy(noisy, labels);
}

struct PathResult
{
    const char* name = "";
    double train_acc = 0.0;
    double test_acc = 0.0;
    double secs_collect_train = 0.0;
    double secs_test = 0.0;
};

static Etalon TrainPath(const char* name, EtalonConfig cfg,
                        std::span<const float> train_fields,
                        std::span<const int> train_labels,
                        PathResult& r)
{
    r.name = name;

    Etalon et(cfg);
    etalon_ex::PrintEtalonHeader(name, et);

    if (!train_fields.empty() && !et.config().bypass_exciter)
    {
        et.Run(train_fields.subspan(0, et.N()));
        double acc_abs = 0.0;
        for (float v : et.LastFeatures())
            acc_abs += std::fabs(static_cast<double>(v));
        acc_abs /= static_cast<double>(et.N());
        std::printf("%s: after one walk, the N bank values average %.4g "
                    "in size (near 0 the field was crushed; around 1 is "
                    "the usual working scale)\n",
                    name, acc_abs);
        std::fflush(stdout);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::printf("%s: collecting %zu fields...\n", name, train_labels.size());
    std::fflush(stdout);
    et.CollectBatch(train_fields, train_labels);

    std::printf("%s: training readout on %zu samples...\n", name, et.NumCollected());
    std::fflush(stdout);
    et.TrainOnCollected();
    r.train_acc = et.AccuracyOnCollected();
    auto t1 = std::chrono::steady_clock::now();
    r.secs_collect_train = std::chrono::duration<double>(t1 - t0).count();
    return et;
}

static void PrintPathSummary(const PathResult& r)
{
    std::printf("%s: train_acc=%.3f test_acc=%.3f  time %.1f+%.1f=%.1fs "
                "(collect+train|test|total)\n",
                r.name, r.train_acc, r.test_acc,
                r.secs_collect_train, r.secs_test,
                r.secs_collect_train + r.secs_test);
    std::fflush(stdout);
}

static void PrintNoiseSweepTable(const char* name, double train_acc,
                                 double collect_train_s,
                                 std::span<const float> sigmas,
                                 std::span<const double> accs,
                                 std::span<const double> secs)
{
    std::printf("%s: train_acc=%.3f  (collect+train %.1fs); "
                "test noise sweep (%zu sigmas)\n",
                name, train_acc, collect_train_s, sigmas.size());
    std::printf("  sigma      test_acc    test_s\n");
    std::printf("  ---------  --------    ------\n");
    for (size_t i = 0; i < sigmas.size(); ++i)
    {
        std::printf("  %9.4f  %8.3f    %6.1f\n",
                    static_cast<double>(sigmas[i]), accs[i], secs[i]);
    }
    std::fflush(stdout);
}

static void PrintSweepCompare(std::span<const float> sigmas,
                              std::span<const double> exciter_accs,
                              std::span<const double> bypass_accs)
{
    std::printf("etalon_mnist: test_acc by sigma (exciter - bypass)\n");
    std::printf("  sigma      exciter     bypass      delta\n");
    std::printf("  ---------  --------    --------    ------\n");
    for (size_t i = 0; i < sigmas.size(); ++i)
    {
        std::printf("  %9.4f  %8.3f    %8.3f    %+.3f\n",
                    static_cast<double>(sigmas[i]),
                    exciter_accs[i], bypass_accs[i],
                    exciter_accs[i] - bypass_accs[i]);
    }
    std::fflush(stdout);
}

/// Score each sigma on a trained instance. Returns clean (σ=0) acc, or -1
/// if the grid has no clean point.
static double RunNoiseSweep(Etalon& et, const char* name,
                            double train_acc, double collect_train_s,
                            std::span<const float> clean_fields,
                            std::span<const int> labels,
                            std::span<const float> grid,
                            std::vector<double>& accs,
                            std::vector<double>& secs)
{
    accs.clear();
    secs.clear();
    accs.reserve(grid.size());
    secs.reserve(grid.size());
    double clean_acc = -1.0;
    for (float sigma : grid)
    {
        auto t0 = std::chrono::steady_clock::now();
        const double acc = ScoreNoisyTest(et, clean_fields, labels, sigma);
        auto t1 = std::chrono::steady_clock::now();
        accs.push_back(acc);
        secs.push_back(std::chrono::duration<double>(t1 - t0).count());
        if (sigma <= 0.0f)
            clean_acc = acc;
    }
    PrintNoiseSweepTable(name, train_acc, collect_train_s, grid, accs, secs);
    return clean_acc;
}

// =============================================================================

int main(int argc, char** argv)
{
    int exit_code = 1;
    try
    {
        const EtalonConfig base = MakeBaseConfig();
        const size_t n_field = size_t{1} << base.exciter.dim;
        if (n_field < static_cast<size_t>(kImgPixels))
        {
            throw std::invalid_argument(
                "etalon_mnist: pack needs N >= 784 (dim >= 10), got N="
                + std::to_string(n_field));
        }

        const char* argv0 = (argc > 0) ? argv[0] : nullptr;
        const auto data_dir = etalon_ex::FindMnistDataDir(argv0);
        const auto data_str =
            std::filesystem::absolute(data_dir).lexically_normal().make_preferred().string();

        std::printf("etalon_mnist: loading IDX from %s\n", data_str.c_str());
        std::fflush(stdout);

        if (kTrainSamples < 0 || kTestSamples < 0)
        {
            throw std::invalid_argument(
                "etalon_mnist: kTrainSamples / kTestSamples must be >= 0 "
                "(0 = whole file)");
        }
        const auto train = etalon_ex::LoadMnist(
            (data_dir / "train-images-idx3-ubyte").string(),
            (data_dir / "train-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTrainSamples));
        const auto test = etalon_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(),
            static_cast<size_t>(kTestSamples));

        const auto emb = etalon_ex::MakeMnistEmbedder(
            static_cast<int>(base.exciter.dim), ToEmbedMode(kPack), kPad);
        if (static_cast<size_t>(emb.capacity()) != n_field)
            throw std::logic_error("embed capacity does not match N");

        const auto plan = emb.plan(kImgSide, kImgSide);
        std::printf("etalon_mnist: pack=%s train=%zu test=%zu "
                    "exciter_epochs=%d bypass_epochs=%d\n",
                    PackModeName(kPack), train.size(), test.size(),
                    kExciterEpochs, kBypassEpochs);
        if (kPack == PackMode::PadLowCenter)
        {
            std::printf(
                "etalon_mnist: PadLowCenter full=%dx%d@ [0,%d) "
                "center=%dx%d@(%d,%d) pattern=%d N=%d\n",
                plan.height_in, plan.width_in, kImgPixels,
                plan.crop_h, plan.crop_w, plan.crop_row0, plan.crop_col0,
                plan.pattern_length, plan.N);
        }
        else
        {
            std::printf("etalon_mnist: PadLow pattern=%d N=%d pad_tail=%d\n",
                        plan.pattern_length, plan.N,
                        plan.N - plan.pattern_length);
        }
        const auto noise_grid = MakeTestNoiseGrid();
        PrintTestNoiseReport(noise_grid);
        std::fflush(stdout);

        std::vector<float> train_fields;
        std::vector<int> train_labels;
        std::vector<float> test_fields;
        std::vector<int> test_labels;
        PackSet(train, emb, train_fields, train_labels);
        PackSet(test, emb, test_fields, test_labels);

        EtalonConfig cfg = base;
        cfg.bypass_exciter = false;
        PathResult exciter;
        Etalon et = TrainPath(
            "etalon_mnist/exciter", cfg,
            train_fields, train_labels, exciter);

        const bool sweep = kTestNoiseSweep;
        bool check_floor = true;
        std::vector<double> exciter_accs;
        std::vector<double> exciter_secs;
        if (sweep)
        {
            const double clean_acc = RunNoiseSweep(
                et, exciter.name, exciter.train_acc, exciter.secs_collect_train,
                test_fields, test_labels, noise_grid,
                exciter_accs, exciter_secs);
            if (clean_acc >= 0.0)
                exciter.test_acc = clean_acc;
            else
                check_floor = false;
        }
        else
        {
            auto t0 = std::chrono::steady_clock::now();
            exciter.test_acc = ScoreNoisyTest(
                et, test_fields, test_labels, 0.0f);
            auto t1 = std::chrono::steady_clock::now();
            exciter.secs_test = std::chrono::duration<double>(t1 - t0).count();
            PrintPathSummary(exciter);
        }

        if (kRunBypass)
        {
            EtalonConfig bypass_cfg = base;
            bypass_cfg.bypass_exciter = true;
            bypass_cfg.readout.epochs = kBypassEpochs;
            PathResult bypass;
            Etalon bypass_et = TrainPath(
                "etalon_mnist/bypass", bypass_cfg,
                train_fields, train_labels, bypass);
            if (sweep)
            {
                std::vector<double> bypass_accs;
                std::vector<double> bypass_secs;
                const double clean_acc = RunNoiseSweep(
                    bypass_et, bypass.name, bypass.train_acc,
                    bypass.secs_collect_train,
                    test_fields, test_labels, noise_grid,
                    bypass_accs, bypass_secs);
                if (clean_acc >= 0.0)
                    bypass.test_acc = clean_acc;
                PrintSweepCompare(noise_grid, exciter_accs, bypass_accs);
            }
            else
            {
                auto t0 = std::chrono::steady_clock::now();
                bypass.test_acc = ScoreNoisyTest(
                    bypass_et, test_fields, test_labels, 0.0f);
                auto t1 = std::chrono::steady_clock::now();
                bypass.secs_test =
                    std::chrono::duration<double>(t1 - t0).count();
                PrintPathSummary(bypass);
                std::printf("etalon_mnist: test_acc delta (exciter - bypass) = %+.3f\n",
                            exciter.test_acc - bypass.test_acc);
                if (bypass.test_acc > exciter.test_acc + 0.05)
                {
                    std::printf("etalon_mnist: note: bypass wins -- packed digits "
                        "are already a readout task; the bank has no "
                        "winning recipe yet\n");
                }
                std::fflush(stdout);
            }
        }

        if (check_floor && exciter.test_acc < kMinExciterTestAcc)
        {
            std::fprintf(stderr,
                         "etalon_mnist: Exciter test accuracy too low "
                         "(need >= %.2f)\n",
                         kMinExciterTestAcc);
        }
        else
        {
            exit_code = 0;
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "etalon_mnist: %s\n", e.what());
        std::fprintf(stderr,
                     "Place uncompressed MNIST IDX files in "
                     "C:\\HypercubeEtalon\\data:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
                     "See examples/README.md (Appendix: MNIST files)\n");
    }
    return exit_code;
}
