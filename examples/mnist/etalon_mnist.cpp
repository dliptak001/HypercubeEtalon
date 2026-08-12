/// @file etalon_mnist.cpp
/// @brief MNIST → pack length-N field → Etalon → held-out test.
///
/// Pipeline: IDX load → balanced subset → PadLowCenter pack → CollectBatch
/// → TrainOnCollected → Accuracy on packed test fields.
///
/// Always scores two paths on the same packed fields:
///   Exciter  — pack → ExciteCube → readout
///   bypass   — pack → readout
///
/// Defaults are a **small balanced subset** (not the 60k recipe). A full
/// dim=10 Exciter pass over 60k is O(N^2 · dim) per image and is gated on
/// a faster bank (see work list item 5).
///
/// Data: C:\HypercubeEtalon\data, then C:\HypercubeWTF\data
/// (see examples/README.md appendix).
///
/// Not ported from wtf_mnist: geometric aug, test-field AWGN, or the noise
/// study write-ups. Those are WTF-orbit questions; rerun them here only after
/// the Exciter scales and cost are settled.

#include "Etalon.h"
#include "find_data_dir.h"
#include "mnist_idx.h"
#include "pack_field.h"
#include "print_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
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
    case PackMode::PadLow:       return hcnn::HCNNSpatialEmbedMode::PadLow;
    case PackMode::PadLowCenter: return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
    }
    return hcnn::HCNNSpatialEmbedMode::PadLowCenter;
}

static const char* PackModeName(PackMode pack)
{
    switch (pack)
    {
    case PackMode::PadLow:       return "PadLow";
    case PackMode::PadLowCenter: return "PadLowCenter";
    }
    return "?";
}

// =============================================================================
// Product knobs
// =============================================================================

static EtalonConfig MakeBaseConfig()
{
    EtalonConfig cfg;

    // PadLow / PadLowCenter need N >= 784 → dim >= 10.
    cfg.exciter.dim = 10;
    cfg.exciter.seed = 13871537636959942979ull;
    // First-guess mixer. Too-small scales collapse the bank toward 0
    // (chance-level readout). These keep the first star O(1):
    // dim=10, |x|<=1 → typical first sum ~ dim * wt * in ≈ 1.
    cfg.exciter.input_scaling = 0.50f;
    cfg.exciter.weight_scaling = 0.20f;

    cfg.train_input_noise_sigma = 0.0f;
    cfg.noise_seed = 1;

    cfg.readout.dim = 0;
    cfg.readout.num_outputs = 10;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.num_layers = 1;
    cfg.readout.use_pooling = true;
    cfg.readout.conv_channels = 16;
    cfg.readout.channel_growth = 1;
    cfg.readout.activation = ReadoutActivation::TANH;
    cfg.readout.epochs = 40;
    cfg.readout.batch_size = 32;
    cfg.readout.lr_max = 0.0015f;
    cfg.readout.lr_min_frac = 0.01f;
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
static constexpr int kTrainPerClass = 100; // 1000 train; set 0 to use file order cap
static constexpr int kTestPerClass = 50;   // 500 test
static constexpr float kPad = -1.0f;
static constexpr int kImgSide = 28;
static constexpr int kImgPixels = kImgSide * kImgSide;
// Health gate is the bypass path (pack + readout). The Exciter recipe is
// still a first guess — do not fail the demo just because the bank loses.
static constexpr double kMinBypassTestAcc = 0.50;
static constexpr size_t kCollectProgress = 64;

// Set false to skip the expensive bank and only run bypass (pack → readout).
static constexpr bool kRunExciter = true;

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

static void CollectWithProgress(Etalon& et,
                                std::span<const float> fields,
                                std::span<const int> labels,
                                const char* name)
{
    const size_t n = et.N();
    const size_t count = labels.size();
    if (count == 0)
        throw std::invalid_argument("CollectWithProgress: empty set");

    for (size_t i = 0; i < count; )
    {
        const size_t take = std::min(kCollectProgress, count - i);
        et.CollectBatch(fields.subspan(i * n, take * n),
                        labels.subspan(i, take));
        i += take;
        std::printf("  %s collected %zu/%zu\n", name, i, count);
        std::fflush(stdout);
    }
}

struct PathResult
{
    const char* name = "";
    double train_acc = 0.0;
    double test_acc = 0.0;
    double secs_collect_train = 0.0;
    double secs_test = 0.0;
};

static PathResult RunPath(const char* name, EtalonConfig cfg,
                          std::span<const float> train_fields,
                          std::span<const int> train_labels,
                          std::span<const float> test_fields,
                          std::span<const int> test_labels,
                          bool progress)
{
    PathResult r;
    r.name = name;

    Etalon et(cfg);
    etalon_ex::PrintEtalonHeader(name, et, cfg);

    if (!train_fields.empty() && !et.BypassExciter())
    {
        et.Run(train_fields.subspan(0, et.N()));
        double acc_abs = 0.0;
        for (float v : et.LastFeatures())
            acc_abs += std::fabs(static_cast<double>(v));
        acc_abs /= static_cast<double>(et.N());
        std::printf("%s: first-map mean|y|=%.4g (bank output scale; "
                    "small is fine if structured)\n",
                    name, acc_abs);
        std::fflush(stdout);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::printf("%s: collecting %zu fields...\n", name, train_labels.size());
    std::fflush(stdout);
    if (progress)
        CollectWithProgress(et, train_fields, train_labels, name);
    else
        et.CollectBatch(train_fields, train_labels);

    std::printf("%s: training readout on %zu samples...\n", name, et.NumCollected());
    std::fflush(stdout);
    et.TrainOnCollected();
    r.train_acc = et.AccuracyOnCollected();
    auto t1 = std::chrono::steady_clock::now();

    std::printf("%s: scoring %zu test fields...\n", name, test_labels.size());
    std::fflush(stdout);
    r.test_acc = et.Accuracy(test_fields, test_labels);
    auto t2 = std::chrono::steady_clock::now();

    r.secs_collect_train = std::chrono::duration<double>(t1 - t0).count();
    r.secs_test = std::chrono::duration<double>(t2 - t1).count();

    std::printf("%s: train_acc=%.3f test_acc=%.3f  time %.1f+%.1f=%.1fs "
                "(collect+train|test|total)\n",
                name, r.train_acc, r.test_acc,
                r.secs_collect_train, r.secs_test,
                r.secs_collect_train + r.secs_test);
    std::fflush(stdout);
    return r;
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

        auto train_all = etalon_ex::LoadMnist(
            (data_dir / "train-images-idx3-ubyte").string(),
            (data_dir / "train-labels-idx1-ubyte").string(), 0);
        auto test_all = etalon_ex::LoadMnist(
            (data_dir / "t10k-images-idx3-ubyte").string(),
            (data_dir / "t10k-labels-idx1-ubyte").string(), 0);

        const auto train = etalon_ex::TakePerClass(train_all, kTrainPerClass);
        const auto test = etalon_ex::TakePerClass(test_all, kTestPerClass);

        const auto emb = etalon_ex::MakeMnistEmbedder(
            static_cast<int>(base.exciter.dim), ToEmbedMode(kPack), kPad);
        if (static_cast<size_t>(emb.capacity()) != n_field)
            throw std::logic_error("embed capacity does not match N");

        const auto plan = emb.plan(kImgSide, kImgSide);
        std::printf("etalon_mnist: pack=%s train=%zu test=%zu epochs=%d\n",
                    PackModeName(kPack), train.size(), test.size(),
                    base.readout.epochs);
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
        std::printf("etalon_mnist: subset is balanced per class; "
                    "not the 60k recipe. Exciter collect is the slow step.\n");
        std::fflush(stdout);

        std::vector<float> train_fields;
        std::vector<int> train_labels;
        std::vector<float> test_fields;
        std::vector<int> test_labels;
        PackSet(train, emb, train_fields, train_labels);
        PackSet(test, emb, test_fields, test_labels);

        PathResult exciter{};
        if (kRunExciter)
        {
            EtalonConfig cfg = base;
            cfg.bypass_exciter = false;
            exciter = RunPath("etalon_mnist/exciter", cfg,
                              train_fields, train_labels,
                              test_fields, test_labels,
                              /*progress=*/true);
        }

        EtalonConfig bypass_cfg = base;
        bypass_cfg.bypass_exciter = true;
        const PathResult bypass = RunPath(
            "etalon_mnist/bypass", bypass_cfg,
            train_fields, train_labels, test_fields, test_labels,
            /*progress=*/false);

        if (kRunExciter)
        {
            std::printf("etalon_mnist: test_acc delta (exciter - bypass) = %+.3f\n",
                        exciter.test_acc - bypass.test_acc);
            if (bypass.test_acc > exciter.test_acc + 0.05)
            {
                std::printf("etalon_mnist: note: bypass wins — packed digits "
                            "are already a readout task; the bank has no "
                            "winning recipe yet\n");
            }
            std::fflush(stdout);
        }

        if (bypass.test_acc < kMinBypassTestAcc)
        {
            std::fprintf(stderr,
                         "etalon_mnist: bypass test accuracy too low "
                         "(need >= %.2f) — pack/readout/data path is broken\n",
                         kMinBypassTestAcc);
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
                     "C:\\HypercubeEtalon\\data or C:\\HypercubeWTF\\data:\n"
                     "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                     "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
                     "See examples/README.md (Appendix: MNIST files)\n");
    }
    return exit_code;
}
