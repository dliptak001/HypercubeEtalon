#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hcnn
{
    class HCNN;
}

/// Which kind of task the readout learns; fixed at construction. Regression
/// predicts continuous values; Classification predicts a discrete class label.
enum class ReadoutTask { Regression, Classification };

/// Activation applied after each Conv layer in the Readout's CNN stack.
/// Mirrors `hcnn::Activation` to keep HCNN.h out of this public header
/// (PIMPL discipline -- mapping lives in Readout.cpp).
enum class ReadoutActivation { TANH, RELU, LEAKY_RELU, NONE };

/// Antipodal pool reduction (only used when @ref ReadoutConfig::use_pooling).
enum class ReadoutPoolType { Max, Avg };

/// Weight-update rule forwarded to HypercubeCNN (default Adam).
enum class ReadoutOptimizer { Adam, Sgd };

/// How @ref Readout::SetState treats optimizer state after loading weights.
/// Eval: restore parameters only (default; safe for inference / export).
/// ResumeTrain: also zero Adam/SGD moments so online training continues cleanly.
enum class ReadoutLoadMode { Eval, ResumeTrain };

/// Cosine-annealing learning-rate schedule: eases the rate smoothly from
/// @p lr_max (at @p progress = 0) down to @p lr_min (at @p progress = 1).
/// Shared between the batch and streaming training paths so the schedule shape
/// is identical. @p progress is clamped to [0, 1].
///
/// Note: batch @ref Readout::Train uses HypercubeCNN's `cosine_lr` (last epoch
/// hits the floor when the horizon is the epoch count). This free function
/// remains for hosts that drive online @ref TrainStep learning rates themselves.
inline float CosineLR(float progress, float lr_max, float lr_min)
{
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(pi * progress));
}

/// Exponential-decay learning-rate schedule: geometric interpolation
/// lr_max * (lr_min/lr_max)^progress, from @p lr_max (at @p progress = 0) down
/// to exactly @p lr_min (at @p progress = 1). Where cosine holds the rate high
/// early and dives late, this drops by a constant *ratio* per unit progress —
/// linear in log-space. Drop-in alternative to @ref CosineLR (same signature
/// and clamping); requires @p lr_max and @p lr_min > 0.
inline float ExponentialDecayLR(float progress, float lr_max, float lr_min)
{
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    return lr_max * std::pow(lr_min / lr_max, progress);
}

/// @brief Architecture and training settings for the @ref Readout CNN.
///
/// The defaults are sensible for a first run; you mainly set @c dim,
/// @c num_outputs, and @c task. The fields split into two groups: the network
/// *shape* (dim, num_outputs, task, num_layers, conv_channels, activation, …)
/// and the *training* hyperparameters (epochs, batch_size, the learning-rate
/// schedule, weight_decay, momentum, seed).
///
/// Must stay trivially copyable (POD) so it can be written into a checkpoint.
struct ReadoutConfig
{
    size_t dim = 0; ///< Input feature dim: features per sample = 2^dim.
    int num_outputs = 1; ///< Classes (classification) or targets (regression).
    ReadoutTask task = ReadoutTask::Regression;
    int num_layers = 1; ///< Conv(+Pool) layers. Default 1 (typical). 0 = auto: min(dim-2, 2).

    /// Append an antipodal pool after each Conv (true = the historical behavior).
    /// The pool pairs each vertex with its bitwise complement, so it mixes *every* bit —
    /// including any block-index bits of a block-structured input. Set false to keep
    /// that structure intact through the conv stack; the flatten readout then sees
    /// twice as many features.
    bool use_pooling = true;
    ReadoutPoolType pool_type = ReadoutPoolType::Max; ///< Used when use_pooling.
    int conv_channels = 16; ///< Base channels for the first conv.
    /// Channel multiplier after each conv stage (historical default: double).
    int channel_growth = 2;
    /// Per-conv batch-norm (HypercubeCNN). Default off — enables BN γ/β + running
    /// stats in the weight blob; keep off for stable checkpoint sizes unless you
    /// intentionally train with BN.
    bool use_batchnorm = false;
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;

    int epochs = 200;
    int batch_size = 32;
    float lr_max = 0.0015f; ///< Cosine annealing peak. Keep <= 0.005 to avoid NaN.
    float lr_min_frac = 0.01f; ///< Floor = lr_max * lr_min_frac.
    int lr_decay_epochs = 0; ///< Cosine decay horizon. 0 = use `epochs`.
    float weight_decay = 0.0f;
    float momentum = 0.9f; ///< SGD momentum (heavy-ball). 0 = plain SGD. Ignored by Adam.
    /// CNN weight-init seed (full 64-bit). Forwarded to HypercubeCNN `weight_seed`.
    /// Seeds that fit in 32 bits keep the historical mt19937 single-arg path so
    /// old campaign inits stay bit-identical; wider seeds expand both halves.
    uint64_t seed = 42;
    ReadoutActivation activation = ReadoutActivation::TANH; ///< Per-Conv-layer activation.

    /// HCNN internal worker-pool size. Forwarded to `hcnn::HCNN`:
    /// 0 = auto (default), 1 = single-threaded (no HCNN background workers),
    /// N > 1 = N workers. Use 1 when the host already parallelizes across ESN
    /// instances (e.g. a multi-seed survey) to avoid nested oversubscription.
    size_t num_threads = 0;

    /// After each batch @ref Readout::Train epoch, score a metric and at the end
    /// restore the best weights seen (regression: min MSE; classification: max
    /// accuracy). Default true (prefer generalization over last-epoch snapshot).
    /// Set false for historical last-epoch behavior. Extra cost: one full forward
    /// over the score set every epoch.
    bool restore_best_epoch = true;
    /// When @c restore_best_epoch is true: fraction of samples (in input order,
    /// taken from the tail) set aside for best-metric selection only — training
    /// uses the prefix. 0 = score the full training set. Clamped to [0, 0.5].
    /// Requires at least 2 samples when > 0.
    float best_epoch_holdout_frac = 0.0f;
};

/// @brief The **trainable half of an @ref ESN**: a small convolutional network
/// (CNN) that maps one reservoir state — N = 2^dim floats — to the task output.
///
/// In an ESN the reservoir is fixed and only the readout learns (see @ref ESN
/// for the whole picture). This class *is* that learner. Each timestep it takes
/// the reservoir's N-number state as its input and produces either a regression
/// vector or class logits, depending on the @ref ReadoutTask chosen at construction.
///
/// ## Data path
/// ```
///   state[N] ──▶ Embed ──▶ [ Conv + Pool ] × L ──▶ Flatten ──▶ Linear ──▶ output
///                          channels grow by channel_growth each layer
/// ```
/// The stack is built via HypercubeCNN's architecture product (`LayerSpec` /
/// `HCNNConfig`) from @c dim: by default L = min(dim - 2, 2) Conv(+Pool) stages
/// (override with @ref ReadoutConfig::num_layers), the first conv using
/// @ref ReadoutConfig::conv_channels channels.
///
/// ## Lifecycle
/// Pick a training path:
///   - **Batch** — collect a set of states, then @ref Train once over all of them.
///   - **Streaming / online** — interleave @ref TrainStep (one state) or
///     @ref TrainStepBatch (a mini-batch) with whatever drives the reservoir.
///
/// Then @ref PredictRaw / @ref PredictClass to use it, and @ref R2 / @ref Accuracy
/// to score it. Save and reload the learned weights with @ref Weights / @ref SetState.
///
/// @note PIMPL: the underlying @c hcnn::HCNN is held by unique_ptr so HCNN.h
///       stays out of this public header (it is included only in Readout.cpp).
class Readout
{
public:
    explicit Readout(const ReadoutConfig& cfg);
    ~Readout();
    Readout(Readout&&) noexcept;
    Readout& operator=(Readout&&) noexcept;

    Readout(const Readout&) = delete;
    Readout& operator=(const Readout&) = delete;

    // ----- Batch training -----

    /// @brief Batch-train (regression): @p targets is num_samples * num_outputs
    /// floats. Continues from current weights — new Readout for a fresh fit.
    /// @throws std::logic_error if task is Classification.
    void Train(const float* states, const float* targets, size_t num_samples);

    /// @brief Batch-train (classification): @p class_labels is num_samples ints
    /// in [0, num_outputs). Continues from current weights.
    /// @throws std::logic_error if task is Regression.
    void Train(const float* states, const int* class_labels, size_t num_samples);

    // ----- Streaming training -----
    //
    // One gradient step at a time, interleaved with whatever drives the
    // reservoir. Overloads are task-typed (no float class indices).

    /// @brief One regression step. @p target is num_outputs floats.
    /// @throws std::logic_error if task is Classification.
    void TrainStep(const float* state, const float* target,
                   float lr, float weight_decay = 0.0f);

    /// @brief One classification step. @p class_label in [0, num_outputs).
    /// @throws std::logic_error if task is Regression.
    void TrainStep(const float* state, int class_label,
                   float lr, float weight_decay = 0.0f);

    /// @brief Regression mini-batch. @p targets is count * num_outputs floats.
    /// @throws std::logic_error if task is Classification.
    void TrainStepBatch(const float* states, const float* targets,
                        size_t count, float lr, float weight_decay = 0.0f);

    /// @brief Classification mini-batch. @p class_labels is count ints.
    /// @throws std::logic_error if task is Regression.
    void TrainStepBatch(const float* states, const int* class_labels,
                        size_t count, float lr, float weight_decay = 0.0f);

    // ----- Prediction -----

    /// @brief Run the network on one @p state and write num_outputs floats to
    /// @p output. Regression: the raw network output. Classification: the raw
    /// class logits (use @ref PredictClass for the argmax label).
    void PredictRaw(const float* state, float* output) const;

    /// @brief Predicted class index — the argmax over the classification logits.
    [[nodiscard]] int PredictClass(const float* state) const;

    // ----- Evaluation -----

    /// @brief R² (coefficient of determination) over @p num_samples (state, target)
    /// pairs, averaged across outputs for multi-output regression. 1.0 is a perfect
    /// fit, 0.0 is no better than always predicting the mean. (Regression metric.)
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// @brief Classification accuracy over @p num_samples (state, label) pairs —
    /// the fraction predicted correctly. Labels are integer class indices.
    /// Multi-class compares argmax to the label; a single output is thresholded
    /// at 0. (Classification metric.)
    [[nodiscard]] double Accuracy(const float* states, const int* labels,
                                  size_t num_samples) const;

    // ----- Accessors -----

    /// @brief Size of one prediction: regression targets, or number of classes.
    [[nodiscard]] size_t NumOutputs() const { return num_outputs_; }
    /// @brief Length of the input state vector the network expects, N = 2^dim.
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    /// @brief Always true — reports that the network exists and has weights worth
    /// persisting (the net_ invariant), not "has been fed training data".
    [[nodiscard]] bool IsTrained() const { return net_ != nullptr; }
    [[nodiscard]] const ReadoutConfig& GetConfig() const { return config_; }

    /// @brief 1-based epoch that produced the restored best weights after the last
    /// @ref Train with @c restore_best_epoch, or 0 if that path was not used / no
    /// snapshot was taken.
    [[nodiscard]] int BestEpoch() const { return best_epoch_; }

    // ----- Serialization -----

    /// @brief Snapshot the live CNN weights as an opaque blob, returned by value so
    /// the copy can't go stale behind a later TrainStep* call (streaming training
    /// mutates the network in place). Pair with @ref SetState.
    ///
    /// Format is an **unversioned** float32 layout promoted to double (same order as
    /// `HCNN::GetWeights`). Prefer @ref SaveHcnnModel for portable, versioned files.
    [[nodiscard]] std::vector<double> Weights() const;

    /// @brief Load a weight blob from @ref Weights back into the network.
    /// An empty blob is ignored.
    /// @param mode Eval (default) restores parameters only; ResumeTrain also resets
    ///        optimizer moments for clean continued training.
    void SetState(std::vector<double> weights,
                  ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// Arch sidecar format version written by @ref SaveHcnnModel (`.arch.json`).
    static constexpr int kArchSidecarVersion = 1;

    /// @brief Write HypercubeCNN-native weights + ESN arch sidecar:
    ///   `@p path_stem.hcnw`     — versioned HCNW (via `hcnn::save_weights`)
    ///   `@p path_stem.arch.json` — architecture knobs + expanded layer list
    /// Pass a path **without** extension (e.g. `"out/readout"`).
    void SaveHcnnModel(const std::string& path_stem) const;

    /// @brief Load `@p path_stem.hcnw` into this readout after validating
    /// `@p path_stem.arch.json` against the live architecture (when the sidecar
    /// exists). If the sidecar is missing, HCNW's own dim/task/layer checks still
    /// apply. @p mode follows @ref SetState.
    void LoadHcnnModel(const std::string& path_stem,
                       ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// @brief Human-readable architecture + parameter count (for logs / demos).
    [[nodiscard]] std::string ArchSummary() const;

private:
    std::unique_ptr<hcnn::HCNN> net_;
    ReadoutConfig config_;
    size_t num_features_ = 0;
    size_t num_outputs_ = 1;
    int best_epoch_ = 0; ///< See @ref BestEpoch.

    void build_architecture();
};
