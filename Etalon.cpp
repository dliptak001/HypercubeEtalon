#include "Etalon.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>

namespace {

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// y[i] = x[i] + N(0, sigma). Deterministic stream from @p salt.
void add_gaussian_noise(const float* x, float* y, size_t n, float sigma,
                        uint64_t salt)
{
    std::mt19937 rng(static_cast<std::uint32_t>(mix64(salt)));
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < n; ++i)
        y[i] = x[i] + dist(rng);
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Etalon::Etalon(const EtalonConfig& cfg)
    : bypass_exciter_(cfg.bypass_exciter),
      train_input_noise_sigma_(cfg.train_input_noise_sigma),
      noise_seed_(cfg.noise_seed),
      exciter_cfg_(cfg.exciter),
      readout_cfg_(cfg.readout)
{
    if (!(train_input_noise_sigma_ >= 0.0f)
        || !std::isfinite(train_input_noise_sigma_))
    {
        throw std::invalid_argument(
            "Etalon: train_input_noise_sigma must be finite and >= 0");
    }

    exciter_ = Exciter::Create(cfg.exciter);
    dim_ = exciter_->Dim();
    n_ = exciter_->Size();

    // Auto-bind readout dim to the Exciter (feature length N = 2^dim).
    if (readout_cfg_.dim == 0)
        readout_cfg_.dim = dim_;
    else if (readout_cfg_.dim != dim_)
    {
        throw std::invalid_argument(
            "Etalon: readout.dim must be 0 (auto) or equal to exciter.dim");
    }

    if (readout_cfg_.num_outputs < 1)
        throw std::invalid_argument("Etalon: readout.num_outputs must be >= 1");

    readout_ = std::make_unique<Readout>(readout_cfg_);
    if (readout_->NumFeatures() != FeatureSize())
    {
        throw std::logic_error(
            "Etalon: readout NumFeatures does not match N = 2^dim");
    }

    field_scratch_.assign(n_, 0.0f);
    last_features_.clear();
    ClearCollected();
}

Etalon::~Etalon() = default;

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

void Etalon::MapInto(std::span<const float> x, std::vector<float>& out_features)
{
    if (x.size() != n_)
    {
        throw std::invalid_argument(
            "Etalon: field size must equal N = 2^dim");
    }

    out_features.resize(n_);

    if (bypass_exciter_)
    {
        std::memcpy(out_features.data(), x.data(), n_ * sizeof(float));
        return;
    }

    // ExciteCube scales in place — never touch the caller's buffer.
    std::memcpy(field_scratch_.data(), x.data(), n_ * sizeof(float));
    const float* y = exciter_->ExciteCube(field_scratch_.data());
    std::memcpy(out_features.data(), y, n_ * sizeof(float));
}

void Etalon::Run(std::span<const float> x)
{
    MapInto(x, last_features_);
}

void Etalon::MapBatchInto(std::span<const float> fields_flat,
                          std::vector<float>& out_features)
{
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Etalon: fields_flat length must be a multiple of N");
    }

    const size_t count = fields_flat.size() / n_;
    out_features.resize(count * n_);
    for (size_t i = 0; i < count; ++i)
    {
        MapInto(fields_flat.subspan(i * n_, n_), last_features_);
        std::memcpy(out_features.data() + i * n_, last_features_.data(),
                    n_ * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Collect
// ---------------------------------------------------------------------------

void Etalon::ClearCollected()
{
    collected_features_.clear();
    collected_labels_.clear();
    collected_targets_.clear();
    num_collected_ = 0;
}

void Etalon::RequireClassification() const
{
    if (readout_cfg_.task != ReadoutTask::Classification)
    {
        throw std::invalid_argument(
            "Etalon: classification API used but task is Regression");
    }
}

void Etalon::RequireRegression() const
{
    if (readout_cfg_.task != ReadoutTask::Regression)
    {
        throw std::invalid_argument(
            "Etalon: regression API used but task is Classification");
    }
}

void Etalon::AppendFeatures(std::span<const float> x)
{
    if (train_input_noise_sigma_ > 0.0f)
    {
        if (noise_field_.size() != n_)
            noise_field_.assign(n_, 0.0f);
        add_gaussian_noise(
            x.data(), noise_field_.data(), n_, train_input_noise_sigma_,
            mix64(noise_seed_ ^ (0x4E4F495300000001ULL + num_collected_)));
        MapInto(noise_field_, last_features_);
    }
    else
    {
        MapInto(x, last_features_);
    }

    const size_t off = num_collected_ * n_;
    collected_features_.resize(off + n_);
    std::memcpy(collected_features_.data() + off, last_features_.data(),
                n_ * sizeof(float));
    ++num_collected_;
}

void Etalon::Collect(std::span<const float> x, int class_label)
{
    RequireClassification();
    if (class_label < 0 || class_label >= readout_cfg_.num_outputs)
    {
        throw std::invalid_argument(
            "Etalon::Collect: class_label must be in [0, num_outputs)");
    }

    AppendFeatures(x);
    collected_labels_.push_back(class_label);
}

void Etalon::Collect(std::span<const float> x, std::span<const float> target)
{
    RequireRegression();
    if (target.size() != static_cast<size_t>(readout_cfg_.num_outputs))
    {
        throw std::invalid_argument(
            "Etalon::Collect: target size must equal num_outputs");
    }

    AppendFeatures(x);
    collected_targets_.insert(collected_targets_.end(), target.begin(),
                              target.end());
}

void Etalon::CollectBatch(std::span<const float> fields_flat,
                          std::span<const int> labels)
{
    RequireClassification();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Etalon::CollectBatch: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    if (labels.size() != count)
    {
        throw std::invalid_argument(
            "Etalon::CollectBatch: labels.size() must equal field count");
    }

    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= readout_cfg_.num_outputs)
        {
            throw std::invalid_argument(
                "Etalon::CollectBatch: label out of range");
        }
        const auto row = fields_flat.subspan(i * n_, n_);
        AppendFeatures(row);
        collected_labels_.push_back(labels[i]);
    }
}

void Etalon::CollectBatch(std::span<const float> fields_flat,
                          std::span<const float> targets_flat)
{
    RequireRegression();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Etalon::CollectBatch: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    const size_t no = static_cast<size_t>(readout_cfg_.num_outputs);
    if (targets_flat.size() != count * no)
    {
        throw std::invalid_argument(
            "Etalon::CollectBatch: targets_flat length must equal "
            "count * num_outputs");
    }

    for (size_t i = 0; i < count; ++i)
    {
        const auto row = fields_flat.subspan(i * n_, n_);
        const auto tgt = targets_flat.subspan(i * no, no);
        AppendFeatures(row);
        collected_targets_.insert(collected_targets_.end(), tgt.begin(),
                                  tgt.end());
    }
}

void Etalon::TrainOnCollected()
{
    if (num_collected_ == 0)
    {
        throw std::invalid_argument(
            "Etalon::TrainOnCollected: no samples collected");
    }

    if (readout_cfg_.task == ReadoutTask::Classification)
    {
        if (collected_labels_.size() != num_collected_)
        {
            throw std::logic_error(
                "Etalon::TrainOnCollected: label buffer size mismatch");
        }
        readout_->Train(collected_features_.data(), collected_labels_.data(),
                        num_collected_);
    }
    else
    {
        const size_t no = static_cast<size_t>(readout_cfg_.num_outputs);
        if (collected_targets_.size() != num_collected_ * no)
        {
            throw std::logic_error(
                "Etalon::TrainOnCollected: target buffer size mismatch");
        }
        readout_->Train(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
    }
}

// ---------------------------------------------------------------------------
// Inference / metrics
// ---------------------------------------------------------------------------

std::vector<float> Etalon::Predict(std::span<const float> x)
{
    Run(x);
    std::vector<float> out(NumOutputs());
    readout_->PredictRaw(last_features_.data(), out.data());
    return out;
}

int Etalon::PredictClass(std::span<const float> x)
{
    RequireClassification();
    Run(x);
    return readout_->PredictClass(last_features_.data());
}

double Etalon::AccuracyOnCollected() const
{
    RequireClassification();
    if (num_collected_ == 0)
        return 0.0;
    return readout_->Accuracy(collected_features_.data(),
                              collected_labels_.data(), num_collected_);
}

double Etalon::R2OnCollected() const
{
    RequireRegression();
    if (num_collected_ == 0)
        return 0.0;
    return readout_->R2(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
}

double Etalon::Accuracy(std::span<const float> fields_flat,
                        std::span<const int> labels)
{
    RequireClassification();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Etalon::Accuracy: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    if (labels.size() != count)
    {
        throw std::invalid_argument(
            "Etalon::Accuracy: labels.size() must equal field count");
    }
    if (count == 0)
        return 0.0;

    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= readout_cfg_.num_outputs)
        {
            throw std::invalid_argument(
                "Etalon::Accuracy: label out of range");
        }
    }

    std::vector<float> feats;
    MapBatchInto(fields_flat, feats);
    return readout_->Accuracy(feats.data(), labels.data(), count);
}

double Etalon::R2(std::span<const float> fields_flat,
                  std::span<const float> targets_flat)
{
    RequireRegression();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Etalon::R2: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    const size_t no = static_cast<size_t>(readout_cfg_.num_outputs);
    if (targets_flat.size() != count * no)
    {
        throw std::invalid_argument(
            "Etalon::R2: targets_flat length must equal count * num_outputs");
    }
    if (count == 0)
        return 0.0;

    std::vector<float> feats;
    MapBatchInto(fields_flat, feats);
    return readout_->R2(feats.data(), targets_flat.data(), count);
}
