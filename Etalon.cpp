#include "Etalon.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>

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
    const uint64_t mixed = mix64(salt);
    const std::uint32_t parts[2] = {
        static_cast<std::uint32_t>(mixed),
        static_cast<std::uint32_t>(mixed >> 32),
    };
    std::seed_seq seq(parts, parts + 2);
    std::mt19937 rng(seq);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < n; ++i)
        y[i] = x[i] + dist(rng);
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Etalon::Etalon(const EtalonConfig& cfg)
    : cfg_(cfg)
{
    if (!(cfg_.train_input_noise_sigma >= 0.0f)
        || !std::isfinite(cfg_.train_input_noise_sigma))
    {
        throw std::invalid_argument(
            "Etalon: train_input_noise_sigma must be finite and >= 0");
    }

    exciter_ = Exciter::Create(cfg_.exciter);
    dim_ = exciter_->Dim();
    n_ = exciter_->N();

    // Auto-bind readout dim to the Exciter (feature length N = 2^dim).
    if (cfg_.readout.dim == 0)
        cfg_.readout.dim = dim_;
    else if (cfg_.readout.dim != dim_)
    {
        throw std::invalid_argument(
            "Etalon: readout.dim must be 0 (auto) or equal to exciter.dim");
    }

    if (cfg_.readout.num_outputs < 1)
        throw std::invalid_argument("Etalon: readout.num_outputs must be >= 1");

    readout_ = std::make_unique<Readout>(cfg_.readout);
    if (readout_->NumFeatures() != n_)
    {
        throw std::logic_error(
            "Etalon: readout NumFeatures does not match N = 2^dim");
    }

    field_scratch_.assign(n_, 0.0f);
    last_features_.clear();
    ClearCollected();
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

void Etalon::MapInto(std::span<const float> x, float* dest)
{
    if (x.size() != n_)
    {
        throw std::invalid_argument(
            "Etalon: field size must equal N = 2^dim");
    }

    if (cfg_.bypass_exciter)
    {
        std::memcpy(dest, x.data(), n_ * sizeof(float));
        return;
    }

    // ExciteCube scales in place — never touch the caller's buffer.
    std::memcpy(field_scratch_.data(), x.data(), n_ * sizeof(float));
    const float* y = exciter_->ExciteCube(field_scratch_.data());
    std::memcpy(dest, y, n_ * sizeof(float));
}

void Etalon::Run(std::span<const float> x)
{
    last_features_.resize(n_);
    MapInto(x, last_features_.data());
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
    if (count == 0)
        return;

    for (size_t i = 0; i < count; ++i)
    {
        MapInto(fields_flat.subspan(i * n_, n_),
                out_features.data() + i * n_);
    }
    last_features_.assign(out_features.data() + (count - 1) * n_,
                          out_features.data() + count * n_);
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
    if (cfg_.readout.task != ReadoutTask::Classification)
    {
        throw std::invalid_argument(
            "Etalon: classification API used but task is Regression");
    }
}

void Etalon::RequireRegression() const
{
    if (cfg_.readout.task != ReadoutTask::Regression)
    {
        throw std::invalid_argument(
            "Etalon: regression API used but task is Classification");
    }
}

void Etalon::MapCollectedOne(std::span<const float> x)
{
    last_features_.resize(n_);
    if (cfg_.train_input_noise_sigma > 0.0f)
    {
        if (noise_field_.size() != n_)
            noise_field_.assign(n_, 0.0f);
        add_gaussian_noise(
            x.data(), noise_field_.data(), n_, cfg_.train_input_noise_sigma,
            mix64(cfg_.noise_seed ^ (0x4E4F495300000001ULL + num_collected_)));
        MapInto(noise_field_, last_features_.data());
    }
    else
    {
        MapInto(x, last_features_.data());
    }
}

void Etalon::MapCollectedBatch(std::span<const float> fields_flat,
                               std::vector<float>& mapped)
{
    if (cfg_.train_input_noise_sigma <= 0.0f)
    {
        MapBatchInto(fields_flat, mapped);
        return;
    }

    const size_t count = fields_flat.size() / n_;
    std::vector<float> noisy(fields_flat.size());
    if (noise_field_.size() != n_)
        noise_field_.assign(n_, 0.0f);
    for (size_t i = 0; i < count; ++i)
    {
        add_gaussian_noise(
            fields_flat.data() + i * n_, noise_field_.data(), n_,
            cfg_.train_input_noise_sigma,
            mix64(cfg_.noise_seed
                  ^ (0x4E4F495300000001ULL + num_collected_ + i)));
        std::memcpy(noisy.data() + i * n_, noise_field_.data(),
                    n_ * sizeof(float));
    }
    MapBatchInto(noisy, mapped);
}

void Etalon::Collect(std::span<const float> x, int class_label)
{
    RequireClassification();
    if (class_label < 0 || class_label >= cfg_.readout.num_outputs)
    {
        throw std::invalid_argument(
            "Etalon::Collect: class_label must be in [0, num_outputs)");
    }

    MapCollectedOne(x);

    const size_t old_feat = collected_features_.size();
    try
    {
        collected_features_.insert(collected_features_.end(),
                                   last_features_.begin(), last_features_.end());
        collected_labels_.push_back(class_label);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_labels_.resize(num_collected_);
        throw;
    }
    ++num_collected_;
}

void Etalon::Collect(std::span<const float> x, std::span<const float> target)
{
    RequireRegression();
    if (target.size() != static_cast<size_t>(cfg_.readout.num_outputs))
    {
        throw std::invalid_argument(
            "Etalon::Collect: target size must equal num_outputs");
    }

    MapCollectedOne(x);

    const size_t old_feat = collected_features_.size();
    const size_t old_tgt = collected_targets_.size();
    try
    {
        collected_features_.insert(collected_features_.end(),
                                   last_features_.begin(), last_features_.end());
        collected_targets_.insert(collected_targets_.end(), target.begin(),
                                  target.end());
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_targets_.resize(old_tgt);
        throw;
    }
    ++num_collected_;
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
        if (labels[i] < 0 || labels[i] >= cfg_.readout.num_outputs)
        {
            throw std::invalid_argument(
                "Etalon::CollectBatch: label out of range");
        }
    }
    if (count == 0)
        return;

    std::vector<float> mapped;
    MapCollectedBatch(fields_flat, mapped);

    const size_t old_feat = collected_features_.size();
    const size_t old_lab = collected_labels_.size();
    try
    {
        collected_features_.reserve(old_feat + mapped.size());
        collected_labels_.reserve(old_lab + count);
        collected_features_.insert(collected_features_.end(), mapped.begin(),
                                   mapped.end());
        collected_labels_.insert(collected_labels_.end(), labels.begin(),
                                 labels.end());
        num_collected_ += count;
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_labels_.resize(old_lab);
        throw;
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
    const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
    if (targets_flat.size() != count * no)
    {
        throw std::invalid_argument(
            "Etalon::CollectBatch: targets_flat length must equal "
            "count * num_outputs");
    }
    if (count == 0)
        return;

    std::vector<float> mapped;
    MapCollectedBatch(fields_flat, mapped);

    const size_t old_feat = collected_features_.size();
    const size_t old_tgt = collected_targets_.size();
    try
    {
        collected_features_.reserve(old_feat + mapped.size());
        collected_targets_.reserve(old_tgt + targets_flat.size());
        collected_features_.insert(collected_features_.end(), mapped.begin(),
                                   mapped.end());
        collected_targets_.insert(collected_targets_.end(),
                                  targets_flat.begin(), targets_flat.end());
        num_collected_ += count;
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_targets_.resize(old_tgt);
        throw;
    }
}

void Etalon::TrainOnCollected()
{
    if (num_collected_ == 0)
    {
        throw std::invalid_argument(
            "Etalon::TrainOnCollected: no samples collected");
    }

    if (cfg_.readout.task == ReadoutTask::Classification)
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
        const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
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
        if (labels[i] < 0 || labels[i] >= cfg_.readout.num_outputs)
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
    const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
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
