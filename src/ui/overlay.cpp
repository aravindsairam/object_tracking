// Drawing helpers for the display window: detection boxes, track boxes
// (per-id colors), the locked target (state-colored box + crosshair + label),
// and the optional magnified zoom panel. Pure rendering — no detection or
// tracking logic lives here.
#include "ot/overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

namespace ot {

namespace {
// Deterministic bright color from a track id (so each id keeps its color).
cv::Scalar color_for_id(int id) {
    const unsigned h = static_cast<unsigned>(id) * 2654435761u;  // Knuth hash
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h % 180, 200, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

void draw_label(cv::Mat& frame, const cv::Rect& r, const std::string& text,
                const cv::Scalar& color) {
    int base = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &base);
    const int ty = std::max(r.y, ts.height + 4);
    cv::rectangle(frame, {r.x, ty - ts.height - 4, ts.width + 6, ts.height + 6},
                  color, cv::FILLED);
    cv::putText(frame, text, {r.x + 3, ty - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {0, 0, 0}, 1, cv::LINE_AA);
}
}  // namespace

void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks,
                 const ClassMap& class_map, bool thin) {
    for (const auto& t : tracks) {
        const cv::Scalar color = thin ? cv::Scalar(140, 140, 140) : color_for_id(t.id);
        const cv::Rect r = t.box.to_rect();
        cv::rectangle(frame, r, color, thin ? 1 : 2);
        if (thin) continue;  // keep labels off the faint background layer

        char buf[96];
        std::snprintf(buf, sizeof(buf), "#%d %s", t.id, class_map.label(t.class_id).c_str());
        draw_label(frame, r, buf, color);
    }
}

void draw_locked_target(cv::Mat& frame, const TargetState& target,
                        const ClassMap& class_map) {
    cv::Scalar color;
    const char* state_txt;
    switch (target.state) {
        case LockState::Locked:   color = {0, 230, 0};   state_txt = "LOCKED";   break;
        case LockState::Coasting: color = {0, 200, 255}; state_txt = "COASTING"; break;
        case LockState::Lost:     color = {0, 0, 255};   state_txt = "LOST";     break;
        default:                  return;  // Acquiring: nothing to draw
    }

    const cv::Rect r = target.box.to_rect();
    cv::rectangle(frame, r, color, 3);

    // Center crosshair.
    const cv::Point c = target.box.center();
    cv::drawMarker(frame, c, color, cv::MARKER_CROSS, 18, 2);

    char buf[128];
    // Show the stable lock id (not the churning MOT id) and the voted class.
    std::snprintf(buf, sizeof(buf), "LOCK #%d %s [%s] %.0f%%", target.lock_id,
                  class_map.label(target.class_id).c_str(), state_txt,
                  target.confidence * 100.0f);
    draw_label(frame, r, buf, color);
}

namespace {
// Border/label color for a lock state (matches draw_locked_target's palette).
cv::Scalar state_color(LockState s) {
    switch (s) {
        case LockState::Locked:   return {0, 230, 0};
        case LockState::Coasting: return {0, 200, 255};
        case LockState::Lost:     return {0, 0, 255};
        default:                  return {120, 120, 120};
    }
}
const char* state_name(LockState s) {
    switch (s) {
        case LockState::Locked:   return "LOCKED";
        case LockState::Coasting: return "COASTING";
        case LockState::Lost:     return "LOST";
        default:                  return "";
    }
}
}  // namespace

cv::Rect zoom_crop_rect(const BBox& box, int W, int H, float scale) {
    if (box.w <= 0.0f || box.h <= 0.0f) return cv::Rect();
    float side = std::max(box.w, box.h) * scale;
    side = std::max(side, 80.0f);                                // floor: keep some context
    side = std::min(side, static_cast<float>(std::min(W, H)));   // never exceed the frame
    const cv::Point2f c = box.center();
    const float half = side * 0.5f;
    // Clamp the center so the full square stays inside the frame (no distortion).
    const float cx = std::clamp(c.x, half, static_cast<float>(W) - half);
    const float cy = std::clamp(c.y, half, static_cast<float>(H) - half);
    int s  = static_cast<int>(std::lround(side));
    s = std::min(s, std::min(W, H));
    int x0 = std::clamp(static_cast<int>(std::lround(cx - half)), 0, W - s);
    int y0 = std::clamp(static_cast<int>(std::lround(cy - half)), 0, H - s);
    return cv::Rect(x0, y0, s, s);
}

cv::Mat render_zoom_panel(const cv::Mat& crop, const cv::Rect& src,
                          const TargetState& target, bool locked,
                          const ClassMap& class_map, int panel_w, int panel_h) {
    cv::Mat panel(panel_h, panel_w, CV_8UC3, cv::Scalar(40, 40, 40));
    const int pad = 12;
    cv::putText(panel, "ZOOM", {pad, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                {220, 220, 220}, 1, cv::LINE_AA);

    // Square magnified image, centered, leaving room below for two label lines.
    const int top = 32, label_block = 64;
    int side = std::min(panel_w - 2 * pad, panel_h - top - label_block);
    side = std::max(side, 32);
    const int ix = (panel_w - side) / 2;
    const int iy = top;
    const cv::Scalar color = state_color(target.state);

    if (locked && !crop.empty()) {
        cv::Mat z;
        cv::resize(crop, z, {side, side}, 0, 0, cv::INTER_LINEAR);
        z.copyTo(panel(cv::Rect(ix, iy, side, side)));
        cv::rectangle(panel, cv::Rect(ix, iy, side, side), color, 2);

        // Outline the exact tracked box within the magnified crop.
        const float k = src.width > 0 ? static_cast<float>(side) / src.width : 0.0f;
        if (k > 0.0f) {
            const cv::Rect b(ix + static_cast<int>((target.box.x - src.x) * k),
                             iy + static_cast<int>((target.box.y - src.y) * k),
                             static_cast<int>(target.box.w * k),
                             static_cast<int>(target.box.h * k));
            cv::rectangle(panel, b, color, 1, cv::LINE_AA);
        }

        char l1[96], l2[96];
        std::snprintf(l1, sizeof(l1), "LOCK #%d  %s", target.lock_id,
                      class_map.label(target.class_id).c_str());
        std::snprintf(l2, sizeof(l2), "%s  %.0f%%  x%.1f", state_name(target.state),
                      target.confidence * 100.0f, k);
        const int ty = iy + side + 26;
        cv::putText(panel, l1, {pad, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {230, 230, 230}, 1, cv::LINE_AA);
        cv::putText(panel, l2, {pad, ty + 24}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv::LINE_AA);
    } else {
        cv::rectangle(panel, cv::Rect(ix, iy, side, side), {80, 80, 80}, 1);
        int base = 0;
        const char* msg = "no target";
        const cv::Size ts = cv::getTextSize(msg, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base);
        cv::putText(panel, msg, {ix + (side - ts.width) / 2, iy + side / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {150, 150, 150}, 1, cv::LINE_AA);
        cv::putText(panel, "click a target to lock", {pad, iy + side + 26},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, {150, 150, 150}, 1, cv::LINE_AA);
    }
    return panel;
}

}  // namespace ot
