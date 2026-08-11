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

/// @brief Fixed recurrent core: N = 2^dim neurons on a Boolean hypercube.
///
/// Hypercube topology, single live state, recurrent neighbor gather only.
/// No learned input weights: staged drive is @c input_scaling * field[v].
///
/// Per-step contract:
/// ```
///   InjectInputField(x, N);
///   Step();
///   // read Outputs()
/// ```
/// Staged input is consumed and zeroed by every @ref Step.
///
/// Non-copyable; obtain instances only via @ref Create.
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

    /// Advance one step: async (in-place) update over vertices 0..N-1, then
    /// clear staged input. Later vertices see already-updated neighbors.
    void Step();

    /// Stage a full length-N input field for the next @ref Step.
    /// @throws std::invalid_argument if @p count != N or @p field is null.
    void InjectInputField(const float* field, size_t count);

    /// Zero dynamical state and staged input. Weights unchanged.
    void Clear();

    [[nodiscard]] const float* Outputs() const { return state_.data(); }

    [[nodiscard]] ReservoirConfig GetConfig() const;

    [[nodiscard]] size_t Dim() const { return dim_; }

    [[nodiscard]] size_t Size() const { return n_; }

    struct Snapshot
    {
        std::vector<float> state; ///< N floats
    };

    [[nodiscard]] Snapshot TakeSnapshot() const;

    void RestoreSnapshot(const Snapshot& snap);

private:
    explicit Reservoir(const ReservoirConfig& cfg);

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t num_weights_ = 0; ///< N · dim recurrent only

    std::vector<float> input_;
    std::vector<float> state_;
    std::vector<float> weight_; ///< recurrent: N · dim

    float input_scaling_ = 0.5f;
    float weight_scaling_ = 0.02f;
    bool verbose_ = false;

    void Initialize();
    void UpdateState(size_t v);
};
