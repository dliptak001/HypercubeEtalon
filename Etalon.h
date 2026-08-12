#pragma once

#include "Exciter.h"
#include "Readout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/// Full product configuration for @ref Etalon.
///
/// Typical knobs: @c exciter.dim / scales / seed, and @c readout.num_outputs /
/// @c readout.task / training hyperparameters. Leave @c readout.dim at 0 to
/// auto-match the Exciter dim (feature length N = 2^dim). Product dim is the
/// Exciter range **[4, 12]**; prefer >= 5 so a pooled readout has room.
struct EtalonConfig
{
    ExciterConfig exciter{};
    ReadoutConfig readout{};

    /// If true, skip the Exciter: readout features are a copy of the length-N
    /// field. Useful as a baseline / ablation. Train-input noise still applies
    /// on collect when σ > 0.
    bool bypass_exciter = false;

    /// Collect-only i.i.d. Gaussian noise on the field before mapping
    /// (σ of N(0,σ) per vertex). 0 = off. Applied on @ref Collect only —
    /// not on @ref Run / @ref Predict / @ref PredictClass. Deterministic from
    /// @ref noise_seed + sample index.
    float train_input_noise_sigma = 0.0f;

    /// Seed for collect-only noise (not the Exciter weight seed).
    uint64_t noise_seed = 1;
};

/// @brief HypercubeEtalon product façade: length-N field → Exciter bank → HCNN.
///
/// ```
///   x[N]  ──▶  [optional train noise]  ──▶  Exciter::ExciteCube  ──▶  y[N]
///                                                                   │
///                                              (or bypass: y = x)   │
///                                                                   ▼
///                                                         Readout (trains)
/// ```
///
/// Lifecycle: @ref Collect → @ref TrainOnCollected → @ref Predict /
/// @ref PredictClass. @ref Run maps a field into @ref LastFeatures without
/// appending to the training set. Save/load, weight blobs, and
/// @ref Readout::IsTrained live on @ref readout(), not here.
///
/// Caller buffers are never mutated: @ref Exciter::ExciteCube scales in place,
/// so every public path copies into internal scratch first.
///
/// One instance is not thread-safe for concurrent public calls.
/// Moved-from objects may only be assigned to or destroyed.
class Etalon
{
public:
    explicit Etalon(const EtalonConfig& cfg);

    Etalon(const Etalon&) = delete;
    Etalon& operator=(const Etalon&) = delete;
    Etalon(Etalon&&) noexcept = default;
    Etalon& operator=(Etalon&&) noexcept = default;

    // ----- Sizes / pieces -----

    [[nodiscard]] size_t Dim() const { return dim_; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }

    /// Resolved product knobs (readout.dim filled in).
    [[nodiscard]] const EtalonConfig& config() const { return cfg_; }
    [[nodiscard]] const Exciter& exciter() const { return *exciter_; }
    /// The trainable head. Weights, HCNW save/load, IsTrained, ArchSummary.
    [[nodiscard]] Readout& readout() { return *readout_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }

    // ----- Map field → features -----

    /// Map one length-N field to features. Does not modify @p x.
    /// Updates @ref LastFeatures. No train-input noise.
    void Run(std::span<const float> x);

    /// Features from the most recent successful map on this instance
    /// (@ref Run, @ref Collect, @ref Predict, @ref PredictClass, @ref Accuracy,
    /// @ref R2, or a batch collect). Valid until the next map on this instance;
    /// copy the values if you need to keep them.
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    // ----- Collect / train -----

    /// Drop all samples collected for batch training.
    void ClearCollected();

    /// Map @p x (with optional collect-only noise), append features + class label.
    /// Updates @ref LastFeatures. Classification task only.
    void Collect(std::span<const float> x, int class_label);

    /// Map @p x (with optional collect-only noise), append features + targets.
    /// @p target length must equal NumOutputs(). Regression task only.
    void Collect(std::span<const float> x, std::span<const float> target);

    /// Bulk collect (classification). @p fields_flat is sample-major, length
    /// count * N; @p labels length count. Serial. Validates all labels before
    /// mapping. On success the last row is left in @ref LastFeatures. A throw
    /// leaves the collected set unchanged.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const int> labels);

    /// Bulk collect (regression). @p targets_flat is sample-major, length
    /// count * NumOutputs(). Serial. On success the last row is left in
    /// @ref LastFeatures. A throw leaves the collected set unchanged.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const float> targets_flat);

    /// Batch-train the readout on all collected samples (requires
    /// NumCollected() > 0). Does not clear the collected set.
    void TrainOnCollected();

    // ----- Inference -----

    /// Fresh map + readout forward; returns NumOutputs() floats.
    /// No train-input noise. Updates @ref LastFeatures.
    [[nodiscard]] std::vector<float> Predict(std::span<const float> x);

    /// Fresh map + argmax class (classification only).
    /// No train-input noise. Updates @ref LastFeatures.
    [[nodiscard]] int PredictClass(std::span<const float> x);

    /// Accuracy on the collected (training) set — not a test-set metric.
    [[nodiscard]] double AccuracyOnCollected() const;

    /// R² on the collected (training) set — not a test-set metric.
    [[nodiscard]] double R2OnCollected() const;

    /// Fresh map (no collect noise) + classification accuracy on a caller-owned
    /// set. @p fields_flat is sample-major, length count * N; @p labels length
    /// count. Updates @ref LastFeatures to the last sample.
    [[nodiscard]] double Accuracy(std::span<const float> fields_flat,
                                  std::span<const int> labels);

    /// Fresh map (no collect noise) + R² on a caller-owned set.
    /// @p targets_flat is sample-major, length count * NumOutputs().
    /// Updates @ref LastFeatures to the last sample.
    [[nodiscard]] double R2(std::span<const float> fields_flat,
                            std::span<const float> targets_flat);

private:
    void MapInto(std::span<const float> x, float* dest);
    void MapBatchInto(std::span<const float> fields_flat,
                      std::vector<float>& out_features);
    void MapCollectedOne(std::span<const float> x);
    void MapCollectedBatch(std::span<const float> fields_flat,
                           std::vector<float>& mapped);
    void RequireClassification() const;
    void RequireRegression() const;

    EtalonConfig cfg_{};
    size_t dim_ = 0;
    size_t n_ = 0;

    std::unique_ptr<Exciter> exciter_;
    std::unique_ptr<Readout> readout_;

    std::vector<float> field_scratch_;   // length N; ExciteCube mutates this
    std::vector<float> last_features_;   // length N
    std::vector<float> noise_field_;     // collect-only when σ > 0

    std::vector<float> collected_features_; // num_collected_ * N
    std::vector<int> collected_labels_;
    std::vector<float> collected_targets_;
    size_t num_collected_ = 0;
};
