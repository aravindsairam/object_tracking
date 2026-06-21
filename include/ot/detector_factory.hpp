#pragma once

#include "ot/config.hpp"
#include "ot/detector.hpp"

#include <memory>

namespace ot {

// Builds the right Detector implementation from config (the C++ analogue of the
// Python model_registry). Dispatches on DetectorCfg::family, and wraps the
// result in a TiledDetector when tiling is enabled. Throws std::runtime_error on
// an unsupported family.
std::unique_ptr<Detector> make_detector(const DetectorCfg& cfg,
                                        const TilingCfg& tiling = {});

}  // namespace ot
