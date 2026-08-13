#include "Etalon.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
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
// Persistent collect thread pool
//
// Background workers live for the Etalon lifetime. Each ForEach is fork-join:
// the calling thread is tid 0; workers 1..nthreads-1 take the other chunks.
// Extra parked workers (pool larger than this job) wait out the generation
// without touching active_.
//
// std::thread + mutex/cv only. No OpenMP.
// Not re-entrant: do not call ForEach from a callback that already runs
// inside ForEach on this pool.
// ---------------------------------------------------------------------------

struct Etalon::CollectPool
{
    explicit CollectPool(size_t background_workers)
    {
        workers_.reserve(background_workers);
        for (size_t i = 0; i < background_workers; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }

    ~CollectPool()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    CollectPool(const CollectPool&) = delete;
    CollectPool& operator=(const CollectPool&) = delete;

    [[nodiscard]] size_t NumThreads() const { return workers_.size() + 1; }

    /// @p func(tid, begin, end) over [0, count). Blocks until all done.
    template <typename F>
    void ForEach(size_t count, size_t nthreads, F&& func)
    {
        if (count == 0)
            return;

        nthreads = std::max<size_t>(1, std::min({nthreads, count, NumThreads()}));
        if (nthreads == 1)
        {
            func(size_t{0}, size_t{0}, count);
            return;
        }

        const size_t chunk = (count + nthreads - 1) / nthreads;
        const int bg = static_cast<int>(nthreads - 1);

        {
            std::lock_guard lock(mutex_);
            exception_ = nullptr;
            job_nthreads_ = nthreads;
            active_.store(bg);
            for_func_ = [&func, chunk, count, nthreads](size_t tid) {
                if (tid >= nthreads)
                    return;
                const size_t b = tid * chunk;
                if (b >= count)
                    return;
                func(tid, b, std::min(b + chunk, count));
            };
            ++generation_;
        }
        cv_work_.notify_all();

        std::exception_ptr caller_ex;
        try
        {
            func(size_t{0}, size_t{0}, std::min(chunk, count));
        }
        catch (...)
        {
            caller_ex = std::current_exception();
        }

        {
            std::unique_lock lock(mutex_);
            cv_done_.wait(lock, [this] { return active_.load() == 0; });
            for_func_ = nullptr;
            if (caller_ex)
            {
                exception_ = nullptr;
                std::rethrow_exception(caller_ex);
            }
            if (exception_)
                std::rethrow_exception(exception_);
        }
    }

private:
    void WorkerLoop(size_t tid)
    {
        size_t local_gen = 0;
        std::function<void(size_t)> fn;
        size_t job_nt = 0;
        while (true)
        {
            {
                std::unique_lock lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ > local_gen; });
                if (stop_)
                    return;
                local_gen = generation_;
                fn = for_func_;
                job_nt = job_nthreads_;
            }

            // Pool may be larger than this job; only tid in [1, job_nt) work.
            if (tid < job_nt && fn)
            {
                try
                {
                    fn(tid);
                }
                catch (...)
                {
                    std::lock_guard elock(mutex_);
                    if (!exception_)
                        exception_ = std::current_exception();
                }

                if (active_.fetch_sub(1) == 1)
                {
                    // Synchronize with ForEach's wait (lost-wakeup guard).
                    { std::lock_guard lock(mutex_); }
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::function<void(size_t)> for_func_;
    std::exception_ptr exception_;
    size_t generation_ = 0;
    size_t job_nthreads_ = 0;
    bool stop_ = false;
    alignas(64) std::atomic<int> active_{0};
};

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

    CollectWorker primary;
    primary.ex = exciter_.get();
    primary.field.assign(n_, 0.0f);
    if (cfg_.train_input_noise_sigma > 0.0f)
        primary.noise.assign(n_, 0.0f);
    collect_workers_.push_back(std::move(primary));
}

Etalon::~Etalon() = default;

void Etalon::RebindPrimaryWorker()
{
    if (collect_workers_.empty())
        return;
    collect_workers_[0].ex = exciter_.get();
    collect_workers_[0].owned.reset();
}

Etalon::Etalon(Etalon&& o) noexcept
    : cfg_(std::move(o.cfg_)),
      dim_(o.dim_),
      n_(o.n_),
      exciter_(std::move(o.exciter_)),
      readout_(std::move(o.readout_)),
      field_scratch_(std::move(o.field_scratch_)),
      last_features_(std::move(o.last_features_)),
      noise_field_(std::move(o.noise_field_)),
      collected_features_(std::move(o.collected_features_)),
      collected_labels_(std::move(o.collected_labels_)),
      collected_targets_(std::move(o.collected_targets_)),
      num_collected_(o.num_collected_),
      collect_workers_(std::move(o.collect_workers_)),
      collect_pool_(std::move(o.collect_pool_))
{
    o.dim_ = 0;
    o.n_ = 0;
    o.num_collected_ = 0;
    RebindPrimaryWorker();
}

Etalon& Etalon::operator=(Etalon&& o) noexcept
{
    if (this == &o)
        return *this;

    collect_pool_.reset();
    collect_workers_.clear();

    cfg_ = std::move(o.cfg_);
    dim_ = o.dim_;
    n_ = o.n_;
    exciter_ = std::move(o.exciter_);
    readout_ = std::move(o.readout_);
    field_scratch_ = std::move(o.field_scratch_);
    last_features_ = std::move(o.last_features_);
    noise_field_ = std::move(o.noise_field_);
    collected_features_ = std::move(o.collected_features_);
    collected_labels_ = std::move(o.collected_labels_);
    collected_targets_ = std::move(o.collected_targets_);
    num_collected_ = o.num_collected_;
    collect_workers_ = std::move(o.collect_workers_);
    collect_pool_ = std::move(o.collect_pool_);

    o.dim_ = 0;
    o.n_ = 0;
    o.num_collected_ = 0;
    RebindPrimaryWorker();
    return *this;
}

// ---------------------------------------------------------------------------
// Collect workers / pool
// ---------------------------------------------------------------------------

size_t Etalon::ResolveCollectThreads(size_t count) const
{
    if (count == 0)
        return 1;
    size_t n = cfg_.collect_threads;
    if (n == 0)
    {
        // Leave headroom for the OS / UI so a long collect does not peg
        // every logical core. Explicit collect_threads = hw still allows
        // a full burn when desired.
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0)
        {
            n = 1;
        }
        else
        {
            const unsigned reserve = (hw >= 8u) ? 2u : 1u;
            n = (hw > reserve) ? static_cast<size_t>(hw - reserve) : 1u;
        }
    }
    if (n < 1)
        n = 1;
    return std::min(n, count);
}

void Etalon::EnsureCollectWorkers(size_t n)
{
    if (n <= collect_workers_.size())
        return;

    collect_workers_.reserve(n);
    while (collect_workers_.size() < n)
    {
        CollectWorker w;
        if (!cfg_.bypass_exciter)
        {
            w.owned = Exciter::Create(cfg_.exciter);
            w.ex = w.owned.get();
        }
        w.field.assign(n_, 0.0f);
        if (cfg_.train_input_noise_sigma > 0.0f)
            w.noise.assign(n_, 0.0f);
        collect_workers_.push_back(std::move(w));
    }
}

void Etalon::EnsureCollectPool(size_t nthreads)
{
    if (nthreads <= 1)
        return;
    const size_t want_bg = nthreads - 1;
    if (!collect_pool_ || collect_pool_->NumThreads() < nthreads)
        collect_pool_ = std::make_unique<CollectPool>(want_bg);
}

void Etalon::PublishLastRow(const float* rows, size_t count)
{
    if (count == 0 || rows == nullptr)
        return;
    const float* last = rows + (count - 1) * n_;
    last_features_.assign(last, last + n_);
}

void Etalon::MapFeaturesParallel(const float* fields_flat, size_t count,
                                 float* dest, bool apply_collect_noise)
{
    if (count == 0)
        return;
    if (fields_flat == nullptr || dest == nullptr)
        throw std::logic_error("Etalon::MapFeaturesParallel: null buffer");

    const size_t nw = ResolveCollectThreads(count);
    EnsureCollectWorkers(nw);
    EnsureCollectPool(nw);

    const bool bypass = cfg_.bypass_exciter;
    const float sigma = cfg_.train_input_noise_sigma;
    const bool do_noise = apply_collect_noise && sigma > 0.0f;
    const size_t noise_base = num_collected_;

    auto run_range = [&](size_t tid, size_t begin, size_t end) {
        CollectWorker& w = collect_workers_[tid];
        for (size_t i = begin; i < end; ++i)
        {
            const float* x = fields_flat + i * n_;
            if (do_noise)
            {
                add_gaussian_noise(
                    x, w.noise.data(), n_, sigma,
                    mix64(cfg_.noise_seed
                          ^ (0x4E4F495300000001ULL + noise_base + i)));
                x = w.noise.data();
            }

            float* out = dest + i * n_;
            if (bypass)
            {
                std::memcpy(out, x, n_ * sizeof(float));
            }
            else
            {
                if (w.ex == nullptr)
                    throw std::logic_error("Etalon::MapFeaturesParallel: null Exciter");
                std::memcpy(w.field.data(), x, n_ * sizeof(float));
                const float* y = w.ex->ExciteCube(w.field.data());
                std::memcpy(out, y, n_ * sizeof(float));
            }
        }
    };

    if (nw <= 1 || !collect_pool_)
        run_range(0, 0, count);
    else
        collect_pool_->ForEach(count, nw, run_range);
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

    MapFeaturesParallel(fields_flat.data(), count, out_features.data(),
                        /*apply_collect_noise=*/false);
    PublishLastRow(out_features.data(), count);
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

    const size_t base = num_collected_;
    const size_t old_feat = collected_features_.size();
    const size_t old_lab = collected_labels_.size();
    try
    {
        collected_labels_.resize(base + count);
        std::memcpy(collected_labels_.data() + base, labels.data(),
                    count * sizeof(int));
        collected_features_.resize((base + count) * n_);
        MapFeaturesParallel(fields_flat.data(), count,
                            collected_features_.data() + base * n_,
                            /*apply_collect_noise=*/true);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_labels_.resize(old_lab);
        throw;
    }
    num_collected_ = base + count;
    PublishLastRow(collected_features_.data() + base * n_, count);
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

    const size_t base = num_collected_;
    const size_t old_feat = collected_features_.size();
    const size_t old_tgt = collected_targets_.size();
    try
    {
        collected_targets_.resize((base + count) * no);
        std::memcpy(collected_targets_.data() + base * no, targets_flat.data(),
                    count * no * sizeof(float));
        collected_features_.resize((base + count) * n_);
        MapFeaturesParallel(fields_flat.data(), count,
                            collected_features_.data() + base * n_,
                            /*apply_collect_noise=*/true);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_targets_.resize(old_tgt);
        throw;
    }
    num_collected_ = base + count;
    PublishLastRow(collected_features_.data() + base * n_, count);
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
