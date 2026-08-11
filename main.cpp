#include "Reservoir.h"

#include <iostream>
#include <vector>

int main() {
    ReservoirConfig cfg;
    cfg.dim = 8;
    cfg.verbose = true;

    auto res = Reservoir::Create(cfg);
    std::vector<float> field(res->Size(), 0.0f);
    field[0] = 1.0f;

    res->InjectInputField(field.data(), field.size());
    res->ExciteCube();

    std::cout << "HypercubeEtalon  N=" << res->Size()
              << "  out[0]=" << res->Outputs()[0] << '\n';
    return 0;
}
