#include "tiled_detector.hpp"

#include <algorithm>

namespace ot {

namespace {
// Tile origins along one axis: step by tile*(1-overlap), last tile flush to edge.
std::vector<int> axis_origins(int length, int tile, float overlap) {
    std::vector<int> origins;
    if (length <= tile) { origins.push_back(0); return origins; }
    const int step = std::max(1, static_cast<int>(std::lround(tile * (1.0f - overlap))));
    for (int o = 0; o + tile < length; o += step) origins.push_back(o);
    origins.push_back(length - tile);  // flush last tile to the right/bottom edge
    // dedupe (the flush tile may coincide with the last stepped one)
    std::sort(origins.begin(), origins.end());
    origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
    return origins;
}

// Intersection-over-Smaller: intersection area / area of the smaller box. A box
// split across a tile seam (a fragment) sits almost entirely inside its full
// counterpart, so IOS ~= 1.0 even though their IOU is low — which is exactly the
// seam-duplicate case plain IOU NMS fails to merge. SAHI's recommended metric.
float ios(const BBox& a, const BBox& b) {
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(a.x + a.w, b.x + b.w);
    const float iy2 = std::min(a.y + a.h, b.y + b.h);
    const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    if (inter <= 0.0f) return 0.0f;
    const float smaller = std::min(a.area(), b.area());
    return smaller > 0.0f ? inter / smaller : 0.0f;
}

// Greedy non-max merge using the IOS metric (SAHI's GREEDYNMM/IOS). Keeps the
// highest-scoring detections first and drops any later, lower-scoring box that
// overlaps a kept SAME-CLASS box by more than `thresh` IOS. Class-aware so a
// person overlapping a car (different classes) can't suppress each other — only
// genuine cross-tile duplicates of the same object are merged.
std::vector<Detection> greedy_ios_merge(std::vector<Detection> dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    kept.reserve(dets.size());
    for (const auto& d : dets) {
        bool dup = false;
        for (const auto& k : kept) {
            if (d.class_id == k.class_id && ios(d.box, k.box) > thresh) { dup = true; break; }
        }
        if (!dup) kept.push_back(d);
    }
    return kept;
}
}  // namespace

TiledDetector::TiledDetector(std::unique_ptr<Detector> base, const TilingCfg& cfg, float nms_iou)
    : base_(std::move(base)), cfg_(cfg), nms_iou_(nms_iou) {}

std::vector<Detection> TiledDetector::detect(const cv::Mat& frame) {
    const int tile = std::min({cfg_.tile, frame.cols, frame.rows});

    // Collect all tile ROIs (+ optional full frame) and their frame-space offsets,
    // then run ONE batched inference over the whole set.
    std::vector<cv::Mat>     imgs;
    std::vector<cv::Point>   offsets;
    for (int y0 : axis_origins(frame.rows, tile, cfg_.overlap)) {
        for (int x0 : axis_origins(frame.cols, tile, cfg_.overlap)) {
            const cv::Rect roi(x0, y0, std::min(tile, frame.cols - x0),
                               std::min(tile, frame.rows - y0));
            imgs.push_back(frame(roi));
            offsets.emplace_back(x0, y0);
        }
    }
    if (cfg_.full_frame) { imgs.push_back(frame); offsets.emplace_back(0, 0); }

    const auto per_image = base_->detect_batch(imgs);

    std::vector<Detection> all;
    for (size_t i = 0; i < per_image.size(); ++i) {
        for (Detection d : per_image[i]) {
            d.box.x += offsets[i].x;
            d.box.y += offsets[i].y;
            all.push_back(d);
        }
    }

    // Merge duplicates from overlapping tiles / full frame with greedy IOS
    // (intersection-over-smaller), class-aware — correctly dedups seam fragments
    // that plain IOU NMS leaves behind, so overlap can be lowered without spawning
    // duplicate boxes. `nms_iou_` is reused as the IOS match threshold.
    return greedy_ios_merge(std::move(all), nms_iou_);
}

}  // namespace ot
