#pragma once

#include "ot/config.hpp"
#include "ot/detector.hpp"

#include <memory>

namespace ot {

// SAHI-style tiled inference. Wraps any base Detector: slices the frame into
// overlapping tiles, runs the base detector on each (so small objects appear at
// a usable scale), shifts the boxes back to frame coordinates, optionally adds a
// whole-frame pass for large objects, and merges everything with NMS.
class TiledDetector : public Detector {
public:
    TiledDetector(std::unique_ptr<Detector> base, const TilingCfg& cfg, float nms_iou);
    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    std::unique_ptr<Detector> base_;
    TilingCfg cfg_;
    float     nms_iou_;
};

}  // namespace ot
