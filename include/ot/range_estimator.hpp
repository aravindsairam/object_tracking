#pragma once

#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/target_state.hpp"
#include "ot/telemetry.hpp"

#include <deque>
#include <unordered_map>

namespace ot {

// One smoothed distance estimate for the locked target.
struct RangeEstimate {
    float       slant_m  = -1.0f;   // camera -> target along the ray [m]
    float       ground_m = -1.0f;   // horizontal distance [m]; <0 = unknown
    RangeMethod method   = RangeMethod::None;
};

// Estimates the distance to the locked target once per inference frame, from
// the working-resolution box plus the frame's telemetry. Methods, best first:
//
//   GroundPlane — cast the pixel ray through the box's BOTTOM-CENTER (the
//   ground contact point), rotate it by the gimbal attitude (yaw is irrelevant
//   for range), intersect the flat ground plane: slant = height / -ray_z.
//   Attitude precedence: set_static_pitch() (--gimbal-pitch) > SRT gb_pitch/
//   gb_roll (enterprise DJI). Needs a positive camera height above the target
//   ground (rel_alt - ground_alt_offset_m). Falls through when the ray misses
//   the ground usefully (above horizon, < min_depression_deg, > max_range_m).
//
//   SizePrior — slant = f_px * prior(class) / max(box.w, box.h), where prior is
//   the class's typical longest dimension (config range.size_priors). No
//   attitude needed; accuracy is prior-dominated (~10-15%), hence the "est" tag.
//
// f_px = equiv_focal_mm / 36 * width. Both f_px and the box live at the WORKING
// resolution, and f_px scales linearly with width, so no native-res rescaling.
// Output is a rolling median over smooth_window frames; the window resets on a
// new lock or a method change (different methods disagree systematically).
//
// Single-threaded use (the inference-thread tail): pure CPU, no locking.
class RangeEstimator {
public:
    RangeEstimator(const RangeCfg& cfg, int work_w, int work_h,
                   const ClassMap& class_map);

    // Fixed gimbal pitch [deg] for the whole clip (0 = level, -90 = nadir);
    // overrides any per-frame SRT attitude. For clips with a known, held pitch.
    void set_static_pitch(float pitch_deg);

    // tele may be null (no SRT record for this frame / no sidecar at all).
    RangeEstimate update(const TargetState& target, const TeleFrame* tele);

    // Forget the smoothing window (lock reset / loop restart).
    void reset();

private:
    RangeEstimate raw_estimate(const TargetState& target, const TeleFrame* tele) const;
    RangeEstimate ground_plane(const TargetState& target, const TeleFrame* tele,
                               double f_px) const;
    double ground_from_slant(double slant_m, const TeleFrame* tele) const;

    RangeCfg cfg_;
    int      width_, height_;
    std::unordered_map<int, float> prior_by_class_;   // class id -> size [m]
    bool     have_static_pitch_ = false;
    float    static_pitch_deg_  = 0.0f;

    int               lock_id_ = -2;                  // window owner
    RangeMethod       window_method_ = RangeMethod::None;
    std::deque<float> slant_win_;
};

}  // namespace ot
