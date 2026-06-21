#pragma once

#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <vector>

namespace ot {

// Multi-object tracker: associates per-frame detections into persistent tracks
// with stable IDs (motion + IoU, ByteTrack). The concrete MOT engine is hidden
// behind a pimpl so its dependency (currently motcpp, AGPL) is isolated to a
// single .cpp — swapping the engine later touches nothing else.
class MotTracker {
public:
    explicit MotTracker(int frame_rate = 30);
    ~MotTracker();

    MotTracker(MotTracker&&) noexcept;
    MotTracker& operator=(MotTracker&&) noexcept;

    // Feeds this frame's detections; returns the currently active tracks.
    std::vector<Track> update(const cv::Mat& frame, const std::vector<Detection>& dets);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ot
