#pragma once

#include "Etalon.h"

/// Raman baseline extraction: dim-11 field in, length-2048 field out.
///
/// Input spectrum and readout target are the same length (`N = 2^11 = 2048`).
/// Task is regression. Other Etalon knobs stay at product defaults until set.
class BaselineExtractor
{
public:
    static constexpr size_t kDim = 11;
    static constexpr size_t kN = size_t{1} << kDim; // 2048

    BaselineExtractor();
    ~BaselineExtractor() = default;

    BaselineExtractor(const BaselineExtractor&) = delete;
    BaselineExtractor& operator=(const BaselineExtractor&) = delete;
    BaselineExtractor(BaselineExtractor&&) noexcept = default;
    BaselineExtractor& operator=(BaselineExtractor&&) noexcept = default;

    [[nodiscard]] size_t Dim() const { return etalon_.Dim(); }
    [[nodiscard]] size_t N() const { return etalon_.N(); }
    [[nodiscard]] size_t NumOutputs() const { return etalon_.NumOutputs(); }

    [[nodiscard]] const EtalonConfig& config() const { return etalon_.config(); }
    [[nodiscard]] Etalon& etalon() { return etalon_; }
    [[nodiscard]] const Etalon& etalon() const { return etalon_; }

private:
    Etalon etalon_;
};
