#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// @brief Construction-time parameters for @ref Reservoir.
struct ReservoirConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[5, 16]**.
    size_t dim = 10;

    /// Master RNG seed for recurrent weight draws.
    uint64_t seed = 7934791766227647176;

    /// Scalar gain on the staged length-N input field (no learned W_in).
    float input_scaling = 0.02f;

    /// Scale on recurrent weight draws: U(-1,1) × weight_scaling.
    float weight_scaling = 0.02f;

    /// If true, print one construction banner.
    bool verbose = false;
};

class Reservoir
{
public:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(new Reservoir(cfg));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    const float* ExciteCube(const float* input_field);

    [[nodiscard]] ReservoirConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t Size() const { return n_; }

private:
    explicit Reservoir(const ReservoirConfig& cfg);

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t num_weights_ = 0; ///< N · dim recurrent only

    std::vector<float> state_;
    std::vector<float> output_;
    std::vector<float> weight_; ///< recurrent: N · dim

    float input_scaling_ = 0.5f;
    float weight_scaling_ = 0.02f;
    bool verbose_ = false;

    void Initialize();
    float ExciteRotation(size_t v_rotation);
};
