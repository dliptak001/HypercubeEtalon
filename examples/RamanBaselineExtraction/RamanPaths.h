#pragma once

#include <string>

inline constexpr const char* kRamanDataRoot =
    "C:/HypercubeEtalon/RamanSpectraLCOHard";
inline constexpr const char* kRamanModelStem =
    "C:/HypercubeEtalon/RamanModels/readout";

inline std::string RamanModelStemFor(bool bypass)
{
    return std::string(kRamanModelStem) + (bypass ? "_bypass" : "_exciter");
}
