#pragma once

#include "ot/target_state.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace ot {

// Predicted search window around the locked target for the ROI fast-path: a
// square centered on the target's next position (current center advanced by one
// frame of its velocity), sized to `scale` x the target's largest dimension. A
// tight crop is fine — the detector stretches it up to the network input, so a
// small/distant target actually gains detail.
//
// The result is ALWAYS a full square fully inside the W x H frame:
//  - in bounds: the caller cuts it with img(roi), which throws on an out-of-frame
//    rect — on the inference thread, so a bad prediction would abort the app.
//  - square: near a border the window SLIDES inward instead of being clipped
//    (same as the zoom panel). A clipped 320x278 crop went through the SAHI tiler
//    as two 278x278 tiles and was stretched non-uniformly; a square crop is one
//    undistorted batch-1 inference.
inline cv::Rect predict_roi(const TargetState& t, float scale, double fps, int W, int H) {
    const float fw = static_cast<float>(W), fh = static_cast<float>(H);

    cv::Point2f c = t.center_px;
    if (!std::isfinite(c.x) || !std::isfinite(c.y)) c = {fw * 0.5f, fh * 0.5f};
    if (fps > 0) {
        const cv::Point2f p(c.x + t.velocity_px_s.x / static_cast<float>(fps),
                            c.y + t.velocity_px_s.y / static_cast<float>(fps));
        if (std::isfinite(p.x) && std::isfinite(p.y)) c = p;   // bad velocity: stay put
    }
    // Clamp the centre into the frame first: also keeps the int casts below defined.
    c.x = std::clamp(c.x, 0.0f, fw);
    c.y = std::clamp(c.y, 0.0f, fh);

    float side = scale * std::max(t.box.w, t.box.h);
    if (!(side >= 320.0f)) side = 320.0f;                       // floor: search margin + context (and NaN)
    const int s = std::max(1, std::min({static_cast<int>(std::min(side, fw + fh)), W, H}));

    // Centre the square on the prediction, then slide it fully inside the frame.
    const int x0 = std::clamp(static_cast<int>(std::lround(c.x)) - s / 2, 0, W - s);
    const int y0 = std::clamp(static_cast<int>(std::lround(c.y)) - s / 2, 0, H - s);
    return cv::Rect(x0, y0, s, s);
}

}  // namespace ot
