#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Etalon.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>

namespace etalon_ex {

inline void PrintExciterConfig(const ExciterConfig& e)
{
    const size_t N = (e.dim < 8 * sizeof(size_t)) ? (size_t{1} << e.dim) : 0;
    const size_t M = (e.subcube_dim >= 1 && e.subcube_dim <= e.dim)
                         ? (size_t{1} << e.subcube_dim)
                         : 0;
    std::printf(
        "exciter: dim=%zu N=%zu subcube_dim=%zu M=%zu seed=%llu in_scale=%.6g wt_scale=%.6g\n",
        e.dim, N, e.subcube_dim, M,
        static_cast<unsigned long long>(e.seed),
        static_cast<double>(e.input_scaling),
        static_cast<double>(e.weight_scaling));
    std::fflush(stdout);
}

/// Live readout knobs after Etalon construction (dim resolved).
/// Does not snapshot Weights() — that copies the whole blob.
inline void PrintReadoutConfig(const Readout& ro)
{
    const ReadoutConfig& r = ro.GetConfig();
    const int d = static_cast<int>(r.dim);
    int layers = (r.num_layers > 0) ? r.num_layers : std::min(d - 2, 2);
    layers = std::max(layers, 1);

    const char* pool_on = r.use_pooling ? "true" : "false";
    const char* pool_type =
        (r.pool_type == ReadoutPoolType::Avg) ? "avg" : "max";

    const char* act = "tanh";
    switch (r.activation)
    {
    case ReadoutActivation::TANH:       act = "tanh"; break;
    case ReadoutActivation::RELU:       act = "relu"; break;
    case ReadoutActivation::LEAKY_RELU: act = "leaky_relu"; break;
    case ReadoutActivation::NONE:       act = "none"; break;
    }

    std::printf(
        "readout: dim=%d layers=%d conv_channels=%d use_pooling=%s pool_type=%s "
        "activation=%s epochs=%d batch=%d lr_max=%.6g\n",
        d, layers, r.conv_channels, pool_on, pool_type, act, r.epochs,
        r.batch_size, static_cast<double>(r.lr_max));
    std::fflush(stdout);
}

inline void PrintEtalonHeader(const char* demo_name, const Etalon& et)
{
    const EtalonConfig& cfg = et.config();
    std::printf("%s: N=%zu dim=%zu bypass_exciter=%s readout_scale=%.6g "
                "collect_threads=%zu%s\n",
                demo_name, et.N(), et.Dim(),
                cfg.bypass_exciter ? "true" : "false",
                static_cast<double>(cfg.readout_scale),
                cfg.collect_threads,
                cfg.collect_threads == 0
                    ? " (auto: leave 1-2 cores free)"
                    : "");
    PrintExciterConfig(cfg.exciter);
    PrintReadoutConfig(et.readout());
    std::fflush(stdout);
}

/// Same probe as the Cascade examples: one packed field through Etalon::Run,
/// then mean |value| at each stage. ~1 is a live field, ~0 is crushed.
/// Etalon keeps no pre-gain buffer, so the Exciter line derives it as
/// features / readout_scale (exact — the gain is a scalar).
inline void ReportStageScales(const char* name, Etalon& et,
                              std::span<const float> packed)
{
    et.Run(packed);
    auto mean_abs = [](std::span<const float> x) {
        if (x.empty())
            return 0.0;
        double a = 0.0;
        for (float v : x)
            a += std::fabs(static_cast<double>(v));
        return a / static_cast<double>(x.size());
    };
    const double f = mean_abs(et.LastFeatures());
    const double scale = static_cast<double>(et.config().readout_scale);
    const bool bypass = et.config().bypass_exciter;
    std::printf("%s: stage scales after one packed train field "
                "(mean |value| over N=%zu; ~1 is a live field, "
                "~0 is crushed)\n",
                name, et.N());
    std::printf("%s:   Input field (packed)                       "
                "mean|x|=%.4g\n",
                name, mean_abs(packed));
    if (!bypass)
    {
        std::printf("%s:   Exciter output (etalon transit)            "
                    "mean|y|=%.4g\n",
                    name, f / scale);
    }
    std::printf("%s:   Readout features (%s * readout_scale) "
                "mean|f|=%.4g\n",
                name, bypass ? "field  " : "Exciter", f);
    std::fflush(stdout);
}

} // namespace etalon_ex
