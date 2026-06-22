#pragma once

#include "ot/config.hpp"
#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <vector>

namespace ot {

class ReidEmbedder;  // appearance embedder, shared with the lock (BoT-SORT only)

// Multi-object tracker: associates per-frame detections into persistent tracks
// with stable IDs. The concrete MOT engine (ByteTrack or BoT-SORT, selected by
// TrackerCfg) is hidden behind a pimpl so its dependency (motcpp, AGPL) stays
// isolated to a single .cpp — swapping the engine later touches nothing else.
class MotTracker {
public:
    // Default ByteTrack (IoU + motion). Preserves prior behavior; used by the probe tools.
    explicit MotTracker(int frame_rate = 30);

    // Config-driven: selects ByteTrack or BoT-SORT and its association tuning. For
    // BoT-SORT with ReID, pass the same embedder the lock uses — it is run on every
    // detection each frame to fuse appearance into association. `reid` may be null
    // (BoT-SORT then runs motion + GMC only, regardless of cfg.with_reid).
    MotTracker(const TrackerCfg& cfg, int frame_rate, std::shared_ptr<ReidEmbedder> reid);

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
