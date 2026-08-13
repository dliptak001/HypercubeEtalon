#include "BaselineExtractor.h"

static EtalonConfig MakeConfig()
{
    EtalonConfig cfg;
    cfg.exciter.dim = BaselineExtractor::kDim;
    cfg.readout.dim = 0; // auto-match Exciter
    cfg.readout.num_outputs = static_cast<int>(BaselineExtractor::kN);
    cfg.readout.task = ReadoutTask::Regression;
    return cfg;
}

BaselineExtractor::BaselineExtractor()
    : etalon_(MakeConfig())
{
}
