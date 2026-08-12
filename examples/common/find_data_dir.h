#pragma once

// MNIST data location for demos. Fixed local deploy roots only — never walks
// the CLion / source-tree clone. Example-only helper.
//
// Search order:
//   1. C:\HypercubeEtalon\data
//   2. C:\HypercubeWTF\data   (same IDX files; avoids a second copy)

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace etalon_ex {

inline bool DirHasMnistIdx(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;
    return fs::is_directory(dir)
           && fs::exists(dir / "train-images-idx3-ubyte")
           && fs::exists(dir / "train-labels-idx1-ubyte")
           && fs::exists(dir / "t10k-images-idx3-ubyte")
           && fs::exists(dir / "t10k-labels-idx1-ubyte");
}

/// Resolve a deploy dir that already contains the four MNIST IDX files.
/// @p argv0 unused (kept for call-site stability).
inline std::filesystem::path FindMnistDataDir(const char* /*argv0*/)
{
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        fs::path("C:/HypercubeEtalon/data"),
        fs::path("C:/HypercubeWTF/data"),
    };

    for (const auto& data : candidates)
    {
        if (DirHasMnistIdx(data))
            return fs::weakly_canonical(data);
    }

    throw std::runtime_error(
        "Cannot find MNIST IDX files.\n"
        "Looked in:\n"
        "  C:\\HypercubeEtalon\\data\n"
        "  C:\\HypercubeWTF\\data\n"
        "Need:\n"
        "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
        "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
        "See examples/README.md (Appendix: MNIST files)");
}

} // namespace etalon_ex
