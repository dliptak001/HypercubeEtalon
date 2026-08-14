#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Etalon.h"

#include <algorithm>
#include <cstdio>

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
        "readout: dim=%d layers=%d use_pooling=%s pool_type=%s activation=%s "
        "epochs=%d batch=%d lr_max=%.6g\n",
        d, layers, pool_on, pool_type, act, r.epochs, r.batch_size,
        static_cast<double>(r.lr_max));
    std::fflush(stdout);
}

inline void PrintEtalonHeader(const char* demo_name, const Etalon& et)
{
    const EtalonConfig& cfg = et.config();
    std::printf("%s: N=%zu dim=%zu bypass_exciter=%s collect_threads=%zu%s\n",
                demo_name, et.N(), et.Dim(),
                cfg.bypass_exciter ? "true" : "false",
                cfg.collect_threads,
                cfg.collect_threads == 0
                    ? " (auto: leave 1-2 cores free)"
                    : "");
    PrintExciterConfig(cfg.exciter);
    PrintReadoutConfig(et.readout());
    std::fflush(stdout);
}

} // namespace etalon_ex
