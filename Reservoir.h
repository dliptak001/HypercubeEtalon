#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


struct ReservoirConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[5, 16]**.
    size_t dim = 10;

    /// Master RNG seed for weight draws (named substreams in Reservoir.cpp).
    uint64_t seed = 7934791766227647176;

    /// Input drive strength. Input weights U(-1,1) then × input_scaling / √dim.
    float input_scaling = 0.02f;

    /// If true, print one construction banner.
    bool verbose = false;
};

/// @brief Fixed recurrent core: N = 2^dim neurons on a Boolean hypercube.
///
/// Hypercube topology, single live state, W_in gather. No delay line, no
/// spectral-radius rescale. Drive via @ref InjectInputField (length N) only.
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
    /// Prefer @ref LoadInitialCondition for episodes.
    void Clear();

    /// Load length-N initial state. Clears staged input.
    /// @throws std::invalid_argument if @p count != N or @p ic is null.
    void LoadInitialCondition(const float* ic, size_t count);

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
    size_t num_input_weights_ = 0;
    size_t num_weights_ = 0;

    std::vector<float> input_;
    std::vector<float> state_;
    std::vector<float> weight_;     ///< [ input: N·dim | recurrent: N·dim ]

    float input_scaling_ = 0.5f;
    bool verbose_ = false;

    void Initialize();
    void UpdateState(size_t v);

    /// Recurrent block starts after the input block.
    [[nodiscard]] size_t RecurrentWeightBase() const { return num_input_weights_; }
};
