#pragma once

#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace ot {

// Family-agnostic object detector. Implementations turn a working-resolution
// frame into people/vehicle detections (already class-filtered and in frame
// pixel coordinates). The tracking layers depend only on this interface.
class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;

    // Detect on several images in one shot. Implementations backed by a batchable
    // model (dynamic batch dim) override this to run a single batched inference
    // (the SAHI tile speed-up). Default: loop detect() per image.
    virtual std::vector<std::vector<Detection>> detect_batch(const std::vector<cv::Mat>& imgs) {
        std::vector<std::vector<Detection>> out;
        out.reserve(imgs.size());
        for (const auto& im : imgs) out.push_back(detect(im));
        return out;
    }
};

}  // namespace ot
