#include "Reservoir.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Reservoir::Reservoir(const ReservoirConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      input_scaling_(cfg.input_scaling),
      weight_scaling_(cfg.weight_scaling),
      verbose_(cfg.verbose)
{
    if (dim_ < 5 || dim_ > 16)
        throw std::invalid_argument("dim must be in 5 <= dim <= 16");

    n_ = 1ULL << dim_;

    // Recurrent weights only: N · dim
    num_weights_ = n_ * dim_;

    input_.assign(n_, 0.0f);
    state_.assign(n_, 0.0f);
    output_.assign(n_, 0.0f);
    weight_.assign(num_weights_, 0.0f);

    Initialize();
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Named substreams (values kept stable vs historical WTF roles where reused).
enum class SeedRole : uint64_t {
    Recurrent = 1
};

// ---------------------------------------------------------------------------
// Weight draw
// ---------------------------------------------------------------------------

void Reservoir::Initialize()
{
    auto seed_for = [this](SeedRole r) {
        return mix64(rng_seed_ ^ (0x100000001B3ULL * static_cast<uint64_t>(r)));
    };
    std::mt19937_64 rng(seed_for(SeedRole::Recurrent));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Clear();

    for (size_t i = 0; i < num_weights_; ++i)
        weight_[i] = static_cast<float>(dist(rng)) * weight_scaling_;

    if (verbose_)
    {
        std::printf("[Reservoir DIM=%zu N=%zu seed=%llu in_scale=%.3g "
                    "w_scale=%.3g]\n",
                    dim_, n_,
                    static_cast<unsigned long long>(rng_seed_),
                    input_scaling_, weight_scaling_);
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

void Reservoir::ComputeOutputs()
{
    for (size_t v = 0; v < n_; ++v)
    {
        // TODO overwrite state_ with scaled input_

        output_[v] = ReflectionPass(v);
    }
}

float Reservoir::ReflectionPass(const size_t v)
{
    float s = 0.0f;
    const float* w = weight_.data() + v * dim_;

    for (size_t j = 0; j < dim_; ++j)
        s += state_[v ^ NearestMask(j)] * w[j];

    return std::tanh(s);
}

// ---------------------------------------------------------------------------
// Drive injection / IC
// ---------------------------------------------------------------------------

void Reservoir::InjectInputField(const float* field, const size_t count)
{
    if (field == nullptr)
        throw std::invalid_argument("InjectInputField: field is null");
    if (count != n_)
        throw std::invalid_argument(
            "InjectInputField: count must equal N = 2^dim");
    std::memcpy(input_.data(), field, n_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// Snapshot / config / clear
// ---------------------------------------------------------------------------

Reservoir::Snapshot Reservoir::TakeSnapshot() const
{
    return Snapshot{state_};
}

void Reservoir::RestoreSnapshot(const Snapshot& snap)
{
    if (snap.state.size() != n_)
        throw std::invalid_argument(
            "RestoreSnapshot: snapshot size does not match this reservoir "
            "(expected state=N)");

    state_ = snap.state;
    std::fill(input_.begin(), input_.end(), 0.0f);
}

ReservoirConfig Reservoir::GetConfig() const
{
    ReservoirConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.input_scaling = input_scaling_;
    cfg.weight_scaling = weight_scaling_;
    cfg.verbose = verbose_;
    return cfg;
}

void Reservoir::Clear()
{
    std::fill(state_.begin(), state_.end(), 0.0f);
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(input_.begin(), input_.end(), 0.0f);
}
