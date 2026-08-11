#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// @brief Construction-time parameters for @ref Exciter.
struct ExciterConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[4, 16]**.
    size_t dim = 10;

    /// Master RNG seed for neighbor weight draws.
    uint64_t seed = 7934791766227647176;

    /// Scalar gain when loading the input field onto the vertices.
    float input_scaling = 0.02f;

    /// Scale on neighbor weight draws: U(-1,1) × weight_scaling.
    float weight_scaling = 0.02f;

    /// If true, print one construction banner.
    bool verbose = false;
};

class Exciter
{
public:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    static std::unique_ptr<Exciter> Create(const ExciterConfig& cfg)
    {
        return std::unique_ptr<Exciter>(new Exciter(cfg));
    }

    Exciter(const Exciter&) = delete;
    Exciter& operator=(const Exciter&) = delete;

    /// Scale @p input_field in place by input_scaling, then for each rotation
    /// reload state from that scaled field, run F/B, write output[r].
    /// @p input_field length N; remains scaled on return.
    /// Returned pointer valid until next ExciteCube/destroy.
    const float* ExciteCube(float* input_field);

    [[nodiscard]] ExciterConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t Size() const { return n_; }

private:
    explicit Exciter(const ExciterConfig& cfg);

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t num_weights_ = 0; ///< N · dim

    std::vector<float> state_;
    std::vector<float> output_;
    std::vector<float> weight_; ///< neighbor: N · dim

    float input_scaling_ = 0.5f;
    float weight_scaling_ = 0.02f;
    bool verbose_ = false;

    void Initialize();
    float ExciteRotation(size_t v_rotation);
};
