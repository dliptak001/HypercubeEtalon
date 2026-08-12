#pragma once

#include "Exciter.h"
#include "Readout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

/// Full product configuration for @ref Etalon.
///
/// Typical knobs: @c exciter.dim / scales / seed, and @c readout.num_outputs /
/// @c readout.task / training hyperparameters. Leave @c readout.dim at 0 to
/// auto-match the Exciter dim (feature length N = 2^dim). Product dim is the
/// Exciter range **[4, 10]**; prefer >= 5 so a pooled readout has room.
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
/// appending to the training set.
///
/// Caller buffers are never mutated: @ref Exciter::ExciteCube scales in place,
/// so every public path copies into internal scratch first.
///
/// One instance is not thread-safe for concurrent public calls.
class Etalon
{
public:
    explicit Etalon(const EtalonConfig& cfg);
    ~Etalon();

    Etalon(const Etalon&) = delete;
    Etalon& operator=(const Etalon&) = delete;

    // ----- Sizes -----

    [[nodiscard]] size_t Dim() const { return dim_; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t FeatureSize() const { return n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }
    [[nodiscard]] bool BypassExciter() const { return bypass_exciter_; }
    [[nodiscard]] float TrainInputNoiseSigma() const { return train_input_noise_sigma_; }
    [[nodiscard]] uint64_t NoiseSeed() const { return noise_seed_; }

    [[nodiscard]] const Exciter& exciter() const { return *exciter_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }
    [[nodiscard]] const ReadoutConfig& readout_config() const { return readout_cfg_; }
    [[nodiscard]] const ExciterConfig& exciter_config() const { return exciter_cfg_; }

    // ----- Map field → features -----

    /// Map one length-N field to features. Does not modify @p x.
    /// Updates @ref LastFeatures. No train-input noise.
    void Run(std::span<const float> x);

    /// Features from the most recent successful @ref Run, serial @ref Collect,
    /// @ref Predict, or @ref PredictClass.
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
    /// count * N; @p labels length count. Serial. Does not update LastFeatures
    /// to a single sample (last row is left in LastFeatures for convenience).
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const int> labels);

    /// Bulk collect (regression). @p targets_flat is sample-major, length
    /// count * NumOutputs(). Serial.
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

    // ----- Readout persistence (thin forwards) -----

    [[nodiscard]] bool IsReadoutTrained() const { return readout_->IsTrained(); }
    [[nodiscard]] std::vector<double> GetReadoutWeights() const
    {
        return readout_->Weights();
    }
    void SetReadoutWeights(std::vector<double> weights,
                           ReadoutLoadMode mode = ReadoutLoadMode::Eval)
    {
        readout_->SetState(std::move(weights), mode);
    }
    void SaveReadoutHcnnModel(const std::string& path_stem) const
    {
        readout_->SaveHcnnModel(path_stem);
    }
    void LoadReadoutHcnnModel(const std::string& path_stem,
                              ReadoutLoadMode mode = ReadoutLoadMode::Eval)
    {
        readout_->LoadHcnnModel(path_stem, mode);
    }
    [[nodiscard]] std::string ReadoutArchSummary() const
    {
        return readout_->ArchSummary();
    }
    [[nodiscard]] int ReadoutBestEpoch() const { return readout_->BestEpoch(); }

private:
    void MapInto(std::span<const float> x, std::vector<float>& out_features);
    void MapBatchInto(std::span<const float> fields_flat,
                      std::vector<float>& out_features);
    void AppendFeatures(std::span<const float> x);
    void RequireClassification() const;
    void RequireRegression() const;

    size_t dim_ = 0;
    size_t n_ = 0;
    bool bypass_exciter_ = false;
    float train_input_noise_sigma_ = 0.0f;
    uint64_t noise_seed_ = 1;

    std::unique_ptr<Exciter> exciter_;
    std::unique_ptr<Readout> readout_;
    ExciterConfig exciter_cfg_{};
    ReadoutConfig readout_cfg_{};

    std::vector<float> field_scratch_;   // length N; ExciteCube mutates this
    std::vector<float> last_features_;   // length N
    std::vector<float> noise_field_;     // collect-only when σ > 0

    std::vector<float> collected_features_; // num_collected_ * N
    std::vector<int> collected_labels_;
    std::vector<float> collected_targets_;
    size_t num_collected_ = 0;
};
