#pragma once

#include "ot/class_map.hpp"
#include "ot/target_state.hpp"
#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace ot {

// Draws tracks with a stable per-id color and "#id label" onto `frame`.
// If `thin` is true, draws faint/thin boxes (used as background under a lock).
void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks,
                 const ClassMap& class_map, bool thin = false);

// --- Lock-reticle animation helpers (pure; unit-tested in test_lock_reticle) --

// True when the lock just (re)engaged: it had no valid target the previous frame
// (Acquiring or Lost) and is Locked now. Coasting<->Locked is NOT an acquire, so
// brief detection gaps don't replay the animation.
bool is_acquire_transition(LockState prev, LockState cur);

// Linear fraction of the acquire animation elapsed, clamped to [0,1]. A duration
// of <= 0 returns 1 (animation already done).
float acquire_progress(double start_s, double now_s, double dur_s);

// Ease-out cubic for t in [0,1] (clamped): fast start, gentle settle.
float ease_out_cubic(float t);

// Corner-bracket arm length for a box (~20% of its shorter side), clamped to a
// readable pixel range.
int bracket_len(const BBox& box);

// Stateful renderer for the locked target: four L-corner brackets + a center
// pip, colored by lock state (green=Locked, amber=Coasting, red=Lost). Each time
// a new lock engages (or a Lost target recovers) it replays a one-shot "acquire"
// animation: the brackets converge from outside the box while the color flashes
// white -> state color. Owns only animation bookkeeping; call draw() once per
// displayed frame with a monotonic `now_s` (seconds).
class LockReticle {
public:
    void draw(cv::Mat& frame, const TargetState& target, const ClassMap& class_map,
              double now_s);

private:
    int       shown_lock_id_  = -1;
    LockState prev_state_      = LockState::Acquiring;
    double    acquire_start_s_ = -1e9;   // far in the past -> no animation in flight
};

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
