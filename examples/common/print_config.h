#pragma once

// Example-only config banners (ASCII for Windows consoles).

#include "Etalon.h"

#include <algorithm>
#include <cstdio>

namespace etalon_ex {

inline void PrintExciterConfig(const ExciterConfig& e)
{
    const size_t N = (e.dim < 8 * sizeof(size_t)) ? (size_t{1} << e.dim) : 0;
    std::printf(
        "exciter: dim=%zu N=%zu seed=%llu in_scale=%.6g wt_scale=%.6g\n",
        e.dim, N,
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

    const char* pool = !r.use_pooling
                           ? "off"
                           : (r.pool_type == ReadoutPoolType::Avg) ? "avg" : "max";

    const char* act = "tanh";
    switch (r.activation)
    {
    case ReadoutActivation::TANH:       act = "tanh"; break;
    case ReadoutActivation::RELU:       act = "relu"; break;
    case ReadoutActivation::LEAKY_RELU: act = "leaky_relu"; break;
    case ReadoutActivation::NONE:       act = "none"; break;
    }

    std::printf(
        "readout: dim=%d layers=%d pooling=%s activation=%s "
        "epochs=%d batch=%d lr_max=%.6g\n",
        d, layers, pool, act, r.epochs, r.batch_size,
        static_cast<double>(r.lr_max));
    std::fflush(stdout);
}

inline void PrintEtalonHeader(const char* demo_name, const Etalon& et,
                              const EtalonConfig& cfg)
{
    std::printf("%s: N=%zu dim=%zu bypass_exciter=%s "
                "train_input_noise_sigma=%.4g noise_seed=%llu\n",
                demo_name, et.N(), et.Dim(),
                et.BypassExciter() ? "true" : "false",
                static_cast<double>(et.TrainInputNoiseSigma()),
                static_cast<unsigned long long>(et.NoiseSeed()));
    PrintExciterConfig(cfg.exciter);
    PrintReadoutConfig(et.readout());
    std::fflush(stdout);
}

} // namespace etalon_ex
