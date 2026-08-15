#pragma once

#include "Exciter.h"
#include "Readout.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/// Configuration for @ref Etalon.
///
/// Typical knobs: @c exciter.dim / scales / seed, and @c readout.num_outputs /
/// @c readout.task / training hyperparameters. Leave @c readout.dim at 0 to
/// auto-match the Exciter dim (feature length N = 2^dim). Dim is the
/// Exciter range **[4, 12]**; prefer >= 5 so a pooled readout has room.
struct EtalonConfig
{
    ExciterConfig exciter{};
    ReadoutConfig readout{};

    /// If true, skip the Exciter: readout features are a copy of the length-N
    /// field. Ablation path; the usual pipeline runs the walk.
    bool bypass_exciter = false;

    /// Parallel workers for bulk @ref Etalon::CollectBatch / @ref Etalon::Accuracy /
    /// @ref Etalon::R2. 0 = auto (leave 1–2 cores free for the OS/UI),
    /// 1 = serial, K = K workers. Single-sample @ref Etalon::Collect / @ref Etalon::Run
    /// is always serial on the primary.
    ///
    /// Auto policy: max(1, hw − 1), or max(1, hw − 2) when hw ≥ 8.
    /// Worker 0 reuses the primary Exciter (no extra weight copy). Extra
    /// workers own clones with the same frozen weights, unless
    /// @c bypass_exciter. A persistent thread pool is kept for the Etalon
    /// lifetime (grows to the high-water mark; does not shrink) so bulk maps
    /// do not re-spawn OS threads each call.
    size_t collect_threads = 0;
};

/// @brief Exciter then Readout: length-N field → features → task outputs.
///
/// ```
///   x[N]  ──▶  Exciter::ExciteCube  ──▶  y[N]  ──▶  Readout
/// ```
///
/// If @c bypass_exciter is set, y is a copy of x.
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
/// Parallelism is internal to bulk @ref CollectBatch / @ref Accuracy / @ref R2.
/// Moved-from objects may only be assigned to or destroyed.
class Etalon
{
public:
    /// @brief Build the Exciter and Readout. Fills @c readout.dim from the
    ///        Exciter when it is 0.
    /// @throws std::invalid_argument if @c readout.dim is neither 0 nor
    ///         equal to @c exciter.dim, or @c readout.num_outputs is < 1.
    ///         Also throws the Exciter and Readout construction checks.
    explicit Etalon(const EtalonConfig& cfg);
    ~Etalon();

    Etalon(const Etalon&) = delete;
    Etalon& operator=(const Etalon&) = delete;
    Etalon(Etalon&&) noexcept;
    Etalon& operator=(Etalon&&) noexcept;

    // ----- Sizes / pieces -----

    [[nodiscard]] size_t Dim() const { return dim_; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    /// Collected feature rows, sample-major, length @ref NumCollected() * N.
    /// Empty when nothing has been collected. Valid until the next
    /// @ref Collect / @ref CollectBatch / @ref ClearCollected.
    [[nodiscard]] std::span<const float> CollectedFeatures() const
    {
        return collected_features_;
    }
    [[nodiscard]] size_t NumOutputs() const { return readout_->NumOutputs(); }
    /// Configured collect-thread preference (0 = auto). Actual workers used on
    /// a given bulk map are min(resolved, sample_count).
    [[nodiscard]] size_t CollectThreads() const { return cfg_.collect_threads; }

    /// Resolved knobs (@c readout.dim filled in).
    [[nodiscard]] const EtalonConfig& config() const { return cfg_; }
    [[nodiscard]] const Exciter& exciter() const { return *exciter_; }
    /// The trainable head. Weights, HCNW save/load, IsTrained, ArchSummary.
    [[nodiscard]] Readout& readout() { return *readout_; }
    [[nodiscard]] const Readout& readout() const { return *readout_; }

    // ----- Map field → features -----

    /// Map one length-N field to features. Does not modify @p x.
    /// Updates @ref LastFeatures.
    /// @throws std::invalid_argument if @p x is not length N.
    void Run(std::span<const float> x);

    /// Features from the most recent completed map on this instance
    /// (@ref Run, @ref Collect, @ref Predict, @ref PredictClass, @ref Accuracy,
    /// @ref R2, or a batch collect). Valid until the next map on this instance;
    /// copy the values if they need to be kept. @ref ClearCollected does not
    /// clear this buffer.
    [[nodiscard]] std::span<const float> LastFeatures() const { return last_features_; }

    // ----- Collect / train -----

    /// Drop all samples collected for batch training.
    /// Does not free collect-worker Exciters or the collect thread pool.
    void ClearCollected();

    /// Map @p x, append features + class label. Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Regression, @p x is not
    ///         length N, or @p class_label is not in [0, num_outputs).
    void Collect(std::span<const float> x, int class_label);

    /// Map @p x, append features + targets. @p target length must equal
    /// NumOutputs(). Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Classification, @p x is
    ///         not length N, or @p target is the wrong length.
    void Collect(std::span<const float> x, std::span<const float> target);

    /// Bulk collect (classification). @p fields_flat is sample-major, length
    /// count * N; @p labels length count. Independent maps fan across
    /// @ref EtalonConfig::collect_threads workers. Validates all labels before
    /// mapping. On success the last row is left in @ref LastFeatures. A throw
    /// leaves the collected set unchanged.
    /// @throws std::invalid_argument if the task is Regression, the buffers
    ///         do not match, or a label is out of range.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const int> labels);

    /// Bulk collect (regression). @p targets_flat is sample-major, length
    /// count * NumOutputs(). Same worker fan-out as the classification
    /// overload. On success the last row is left in @ref LastFeatures. A throw
    /// leaves the collected set unchanged.
    /// @throws std::invalid_argument if the task is Classification or the
    ///         buffers do not match.
    void CollectBatch(std::span<const float> fields_flat,
                      std::span<const float> targets_flat);

    /// Batch-train the readout on all collected samples. Continues from
    /// the current readout weights. Does not clear the collected set.
    /// @throws std::invalid_argument if nothing has been collected.
    void TrainOnCollected();

    // ----- Inference -----

    /// Fresh map + readout forward; returns NumOutputs() floats.
    /// Updates @ref LastFeatures.
    /// @throws std::invalid_argument if @p x is not length N.
    [[nodiscard]] std::vector<float> Predict(std::span<const float> x);

    /// Fresh map + argmax class (classification only).
    /// Updates @ref LastFeatures.
    /// @throws std::invalid_argument if the task is Regression or @p x is
    ///         not length N.
    [[nodiscard]] int PredictClass(std::span<const float> x);

    /// Accuracy on the collected (training) set — not a test-set metric.
    /// Empty set is 0. @throws std::invalid_argument if the task is Regression.
    [[nodiscard]] double AccuracyOnCollected() const;

    /// R² on the collected (training) set — not a test-set metric.
    /// Empty set is 0. @throws std::invalid_argument if the task is Classification.
    [[nodiscard]] double R2OnCollected() const;

    /// Fresh map + classification accuracy on a caller-owned set.
    /// @p fields_flat is sample-major, length count * N; @p labels length
    /// count. Updates @ref LastFeatures to the last sample. Empty set is 0.
    /// @throws std::invalid_argument if the task is Regression, the buffers
    ///         do not match, or a label is out of range.
    [[nodiscard]] double Accuracy(std::span<const float> fields_flat,
                                  std::span<const int> labels);

    /// Fresh map + R² on a caller-owned set.
    /// @p targets_flat is sample-major, length count * NumOutputs().
    /// Updates @ref LastFeatures to the last sample. Empty set is 0.
    /// @throws std::invalid_argument if the task is Classification or the
    ///         buffers do not match.
    [[nodiscard]] double R2(std::span<const float> fields_flat,
                            std::span<const float> targets_flat);

private:
    /// One map runner. Worker 0 aliases the primary Exciter (no second
    /// weight copy). Workers 1.. use owned clones.
    struct CollectWorker
    {
        Exciter* ex = nullptr;            // non-owning view
        std::unique_ptr<Exciter> owned;   // null for primary alias
        std::vector<float> field;         // length N; ExciteCube mutates this
    };

    /// Persistent fork-join pool (see Etalon.cpp).
    struct CollectPool;

    void MapInto(std::span<const float> x, float* dest);
    void MapBatchInto(std::span<const float> fields_flat,
                      std::vector<float>& out_features);
    void MapCollectedOne(std::span<const float> x);
    void MapFeaturesParallel(const float* fields_flat, size_t count,
                             float* dest);
    void RequireClassification() const;
    void RequireRegression() const;

    [[nodiscard]] size_t ResolveCollectThreads(size_t count) const;
    void EnsureCollectWorkers(size_t n);
    void EnsureCollectPool(size_t nthreads);
    void RebindPrimaryWorker();
    void PublishLastRow(const float* rows, size_t count);

    EtalonConfig cfg_{};
    size_t dim_ = 0;
    size_t n_ = 0;

    std::unique_ptr<Exciter> exciter_;
    std::unique_ptr<Readout> readout_;

    std::vector<float> field_scratch_;   // length N; serial ExciteCube scratch
    std::vector<float> last_features_;   // length N

    std::vector<float> collected_features_; // num_collected_ * N
    std::vector<int> collected_labels_;
    std::vector<float> collected_targets_;
    size_t num_collected_ = 0;

    std::vector<CollectWorker> collect_workers_;
    std::unique_ptr<CollectPool> collect_pool_;
};
