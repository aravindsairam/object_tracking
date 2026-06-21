#pragma once

#include "ot/class_map.hpp"
#include "ot/target_state.hpp"
#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace ot {

// Draws detection boxes + "label score" onto `frame` in place.
void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets,
                     const ClassMap& class_map);

// Draws tracks with a stable per-id color and "#id label" onto `frame`.
// If `thin` is true, draws faint/thin boxes (used as background under a lock).
void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks,
                 const ClassMap& class_map, bool thin = false);

// Draws the locked target prominently: thick box + center crosshair + label,
// colored by lock state (green=Locked, amber=Coasting, red=Lost).
void draw_locked_target(cv::Mat& frame, const TargetState& target,
                        const ClassMap& class_map);

// Square crop rect around `box` (with `scale`x context margin so the object
// fills ~1/scale of the crop), clamped to stay fully inside the WxH frame.
// Returns a valid square rect for any box with positive area; an empty rect for
// an empty box. Used to grab the un-annotated source region for the zoom panel.
cv::Rect zoom_crop_rect(const BBox& box, int W, int H, float scale = 2.2f);

// Builds a (panel_w x panel_h) side-panel image showing a magnified view of the
// locked target. `crop` is a clean (un-annotated) sub-image of the frame around
// the target and `src` is its rect in frame coords; pass an empty Mat/rect when
// nothing is locked (a "no target" placeholder is drawn). The panel border, the
// inner box outline, and the label are colored by lock state.
cv::Mat render_zoom_panel(const cv::Mat& crop, const cv::Rect& src,
                          const TargetState& target, bool locked,
                          const ClassMap& class_map, int panel_w, int panel_h);

}  // namespace ot
