#include "ot/detector_factory.hpp"

#include "rfdetr_detector.hpp"
#include "tiled_detector.hpp"
#include "yolo_detector.hpp"

#include <stdexcept>

namespace ot {

namespace {
std::unique_ptr<Detector> make_base(const DetectorCfg& cfg) {
    // YOLO, WALDO and UniDrone all share the YOLO output family (raw head or
    // NMS-embedded), so one decoder serves them; only the class_map differs.
    if (cfg.family == "yolo" || cfg.family == "waldo" || cfg.family == "unidrone") {
        return std::make_unique<YoloDetector>(cfg);
    }
    if (cfg.family == "rfdetr") {
        return std::make_unique<RfDetrDetector>(cfg);
    }
    throw std::runtime_error("make_detector: unknown family '" + cfg.family + "'");
}
}  // namespace

std::unique_ptr<Detector> make_detector(const DetectorCfg& cfg, const TilingCfg& tiling) {
    auto base = make_base(cfg);
    if (tiling.enabled) {
        return std::make_unique<TiledDetector>(std::move(base), tiling, cfg.nms_iou);
    }
    return base;
}

}  // namespace ot
