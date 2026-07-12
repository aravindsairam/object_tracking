#pragma once

#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <cstdint>

namespace ot {

// How the locked target's distance was obtained (see ot::RangeEstimator).
enum class RangeMethod : uint8_t {
    None        = 0,   // no estimate this frame
    GroundPlane = 1,   // pixel ray x ground plane, needs gimbal attitude ("gp")
    SizePrior   = 2,   // typical class size vs box size, no attitude ("est")
};

// The per-frame result of the lock: where the locked target is and how much we
// trust it. This is the structured output a future control loop would consume.
struct TargetState {
    int64_t     frame_index = -1;
    double      timestamp_s = 0.0;
    LockState   state = LockState::Acquiring;
    BBox        box;                      // measured (Locked) or predicted (Coasting)
    cv::Point2f center_px{0, 0};
    cv::Point2f center_norm{0, 0};        // center / frame size, in [0,1]
    cv::Point2f velocity_px_s{0, 0};
    float       confidence = 0.0f;        // system-level lock confidence in [0,1]
    int         class_id = -1;            // stabilized (voted) class of the target
    int         track_id = -1;            // raw MOT track id (churns across re-acquires)
    int         lock_id  = -1;            // stable id for THIS lock, kept across re-acquires

    // Approximate distance to the target (smoothed; see ot::RangeEstimator).
    float       range_slant_m  = -1.0f;   // camera -> target along the ray [m]
    float       range_ground_m = -1.0f;   // horizontal distance [m]; <0 = unknown
    RangeMethod range_method   = RangeMethod::None;

    bool valid() const { return state == LockState::Locked || state == LockState::Coasting; }
};

}  // namespace ot
