#include "Exciter.h"

#include <cmath>
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
      weight_scaling_(cfg.weight_scaling)
{
    if (dim_ < 4 || dim_ > 10)
        throw std::invalid_argument("Exciter: dim must be in 4 <= dim <= 10");

    n_ = 1ULL << dim_;
    num_weights_ = n_ * dim_;

    state_.assign(n_, 0.0f);
    output_.assign(n_, 0.0f);
    weight_.assign(num_weights_, 0.0f);

    Initialize();
}

// ---------------------------------------------------------------------------
// Weight draw
// ---------------------------------------------------------------------------

void Exciter::Initialize()
{
    std::mt19937_64 rng(rng_seed_);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < num_weights_; ++i)
        weight_[i] = static_cast<float>(dist(rng)) * weight_scaling_;
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

const float* Exciter::ExciteCube(float* input_field)
{
    if (input_field == nullptr)
        throw std::invalid_argument("ExciteCube: input_field is null");

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

    // Forward: full neighbor star, visit v = 0 .. N-1 (ends at antipode of r).
    for (int64_t v = 0; v < n; ++v)
    {
        const size_t vv = static_cast<size_t>(v) ^ v_rotation;
        float s = 0.0f;
        const float* w = weight_.data() + vv * dim_;
        for (size_t j = 0; j < dim_; ++j)
            s += state_[vv ^ NearestMask(j)] * w[j];

        state_[vv] = std::tanh(s);
    }

    // Reverse turnaround: skip v = N-1 (since antipode was just written at end of forward).
    for (int64_t v = n - 2; v >= 0; --v)
    {
        const size_t vv = static_cast<size_t>(v) ^ v_rotation;
        float s = 0.0f;
        const float* w = weight_.data() + vv * dim_;
        for (size_t j = 0; j < dim_; ++j)
            s += state_[vv ^ NearestMask(j)] * w[j];

        state_[vv] = std::tanh(s);
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
    return cfg;
}
