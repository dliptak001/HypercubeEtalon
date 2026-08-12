#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// @brief Construction-time parameters for @ref Exciter.
struct ExciterConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[4, 12]**:
    /// 4 is the smallest cube with a useful neighbor star; 12 is the cost cap
    /// (N = 4096). Default walk is a half-cube (`s = 1`).
    size_t dim = 8;

    /// Master RNG seed for neighbor weight draws.
    uint64_t seed = 7934791766227647176;

    /// Scalar gain applied once in place to the input field in @ref ExciteCube.
    float input_scaling = 0.02f;

    /// Scale on neighbor weight draws: U(-1,1) × weight_scaling.
    float weight_scaling = 0.02f;

    /// Pin this many high bits: bounce on a (dim-s)-face through r.
    /// 0 = whole cube, 1 = half (default), 2 = quarter, …
    /// Valid range **[0, dim)**, so the walk has at least two corners
    /// (M = N >> s >= 2). Full star: off-face neighbors stay at the
    /// scaled input.
    size_t s = 1;
};

/// Hypercube field exciter: fixed neighbor weights, XOR-rotated F/B sweeps.
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

    /// Scale @p input_field in place by input_scaling; per rotation reload
    /// state, run F/B, write output[r]. Length must be Size(); non-null;
    /// must not alias internals. Re-calling on same buffer scales again.
    /// @return length-N output; valid until next ExciteCube or destroy.
    [[nodiscard]] const float* ExciteCube(float* input_field);

    [[nodiscard]] ExciterConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t Size() const { return n_; }

    [[nodiscard]] size_t S() const { return s_; }

    /// Corners visited per bounce: M = N >> s.
    [[nodiscard]] size_t WalkSize() const { return m_; }

private:
    explicit Exciter(const ExciterConfig& cfg);

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t s_ = 0;
    size_t m_ = 0; ///< N >> s
    size_t num_weights_ = 0; ///< N · dim

    std::vector<float> state_;
    std::vector<float> output_;
    std::vector<float> weight_; ///< neighbor: N · dim

    float input_scaling_ = 0.02f;
    float weight_scaling_ = 0.02f;

    void Initialize();
    void UpdateSite(size_t vv);
    float ExciteRotation(size_t r);
};
