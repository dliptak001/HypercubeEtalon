#include "Exciter.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Exciter::Exciter(const ExciterConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      input_scaling_(cfg.input_scaling),
      weight_scaling_(cfg.weight_scaling),
      verbose_(cfg.verbose)
{
    if (dim_ < 4 || dim_ > 16)
        throw std::invalid_argument("dim must be in 4 <= dim <= 16");

    n_ = 1ULL << dim_;

    // Neighbor weights: N · dim
    num_weights_ = n_ * dim_;

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

enum class SeedRole : uint64_t
{
    Weights = 1
};

// ---------------------------------------------------------------------------
// Weight draw
// ---------------------------------------------------------------------------

void Exciter::Initialize()
{
    auto seed_for = [this](SeedRole r)
    {
        return mix64(rng_seed_ ^ (0x100000001B3ULL * static_cast<uint64_t>(r)));
    };
    std::mt19937_64 rng(seed_for(SeedRole::Weights));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < num_weights_; ++i)
        weight_[i] = static_cast<float>(dist(rng)) * weight_scaling_;

    if (verbose_)
    {
        std::printf("[Exciter DIM=%zu N=%zu seed=%llu in_scale=%.3g "
                    "w_scale=%.3g]\n",
                    dim_, n_,
                    static_cast<unsigned long long>(rng_seed_),
                    input_scaling_, weight_scaling_);
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

const float* Exciter::ExciteCube(float* input_field)
{
    // Scale once; every rotation reloads this same IC into state_.
    for (size_t i = 0; i < n_; ++i)
        input_field[i] *= input_scaling_;

    for (size_t v = 0; v < n_; ++v)
    {
        std::memcpy(state_.data(), input_field, n_ * sizeof(float));
        output_[v] = ExciteRotation(v);
    }

    return output_.data();
}

float Exciter::ExciteRotation(const size_t v_rotation)
{
    const auto n = static_cast<int64_t>(n_);

    // forward
    for (int64_t v = 0; v < n; ++v)
    {
        const size_t vv = v ^ v_rotation;
        float s = 0.0f;
        const float* w = weight_.data() + vv * dim_;
        for (size_t j = 1; j < dim_; ++j)
            s += state_[vv ^ NearestMask(j)] * w[j];

        state_[vv] = std::tan(s);
    }

    // reverse
    for (int64_t v = n - 1; v >= 0; --v)
    {
        const size_t vv = v ^ v_rotation;
        float s = 0.0f;
        const float* w = weight_.data() + vv * dim_;
        for (size_t j = 0; j < dim_; ++j)
            s += state_[vv ^ NearestMask(j)] * w[j];

        state_[vv] = std::tan(s);
    }

    return state_[v_rotation];
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

ExciterConfig Exciter::GetConfig() const
{
    ExciterConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.input_scaling = input_scaling_;
    cfg.weight_scaling = weight_scaling_;
    cfg.verbose = verbose_;
    return cfg;
}
