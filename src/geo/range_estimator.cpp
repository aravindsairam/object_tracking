// Locked-target distance estimation. See range_estimator.hpp for the method
// chain and docs/2d_distance/range_design.md for the derivation and error
// budget (size prior ~10-15%, dominated by how typical the class prior is).
#include "ot/range_estimator.hpp"

#include "ot/geodesy.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ot {

namespace {
// Upper median of a small window (copies; windows are ~15 elements).
float median(const std::deque<float>& w) {
    std::vector<float> v(w.begin(), w.end());
    const auto mid = v.begin() + v.size() / 2;
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

constexpr double kDeg = kDegToRad;

// R such that v_world = R * v_cam; camera frame +x right, +y down, +z forward;
// world x East, y North, z Up. yaw = optical-axis heading CW from north, pitch
// 0 = level / -90 = nadir, roll about the optical axis. Same verified form as
// the feat/2d-geolocation reference. Range only needs the vertical component
// (yaw = 0); the target position passes the camera heading.
void rotation(double yaw_deg, double pitch_deg, double roll_deg, double R[3][3]) {
    const double psi = yaw_deg * kDeg, th = pitch_deg * kDeg, ph = roll_deg * kDeg;
    const double fwd[3]    = {std::sin(psi) * std::cos(th),
                              std::cos(psi) * std::cos(th),
                              std::sin(th)};
    const double right0[3] = {std::cos(psi), -std::sin(psi), 0.0};
    const double down0[3]  = {fwd[1] * right0[2] - fwd[2] * right0[1],
                              fwd[2] * right0[0] - fwd[0] * right0[2],
                              fwd[0] * right0[1] - fwd[1] * right0[0]};
    double right[3], down[3];
    for (int i = 0; i < 3; ++i)
        right[i] = right0[i] * std::cos(ph) + down0[i] * std::sin(ph);
    down[0] = fwd[1] * right[2] - fwd[2] * right[1];
    down[1] = fwd[2] * right[0] - fwd[0] * right[2];
    down[2] = fwd[0] * right[1] - fwd[1] * right[0];
    for (int i = 0; i < 3; ++i) {
        R[i][0] = right[i];
        R[i][1] = down[i];
        R[i][2] = fwd[i];
    }
}
}  // namespace

RangeEstimator::RangeEstimator(const RangeCfg& cfg, int work_w, int work_h,
                               const ClassMap& class_map)
    : cfg_(cfg), width_(work_w), height_(work_h) {
    // Resolve the name-keyed config priors to this model's class ids once, so
    // the per-frame path is a plain int lookup.
    for (const int id : class_map.kept_ids()) {
        const auto it = cfg_.size_priors.find(class_map.label(id));
        if (it != cfg_.size_priors.end() && it->second > 0.0f)
            prior_by_class_[id] = it->second;
    }
}

void RangeEstimator::set_static_pitch(float pitch_deg) {
    have_static_pitch_ = true;
    static_pitch_deg_ = pitch_deg;
}

void RangeEstimator::reset() {
    lock_id_ = -2;
    window_method_ = RangeMethod::None;
    slant_win_.clear();
}

double RangeEstimator::ground_from_slant(double slant_m, const TeleFrame* tele) const {
    // Flat-ground geometry: ground = sqrt(slant^2 - height^2). Holds for both
    // methods; unknown (<0) without a usable camera height above the target.
    // slant == height is a nadir fix (ground 0), slant < height means the
    // size-prior range contradicts the altitude — report the ground as unknown.
    if (!tele) return -1.0;
    const double h = tele->rel_alt - cfg_.ground_alt_offset_m;
    if (h <= 0.0 || slant_m < h) return -1.0;
    return std::sqrt(std::max(0.0, slant_m * slant_m - h * h));
}

double RangeEstimator::focal_mm(const TeleFrame* tele) const {
    return cfg_.equiv_focal_mm > 0.0f ? cfg_.equiv_focal_mm
         : (tele && tele->has_focal) ? tele->focal_len : 0.0;
}

// Attitude precedence: static --gimbal-pitch > per-frame telemetry gb_*.
// Leaves pitch/roll at 0 (level) and returns false when neither exists.
bool RangeEstimator::attitude(const TeleFrame* tele, double& pitch, double& roll) const {
    pitch = 0.0;
    roll = 0.0;
    if (have_static_pitch_) {
        pitch = static_pitch_deg_;
        return true;
    }
    if (tele && tele->has_attitude) {
        pitch = tele->gb_pitch;
        roll = tele->gb_roll;
        return true;
    }
    return false;
}

// Unit ray through the box's ground-contact pixel (bottom-center), world frame.
void RangeEstimator::world_ray(const TargetState& t, double f_px, double yaw_deg,
                               double pitch_deg, double roll_deg, double dw[3]) const {
    const double u = t.box.x + t.box.w * 0.5;
    const double v = t.box.y + t.box.h;
    double d[3] = {(u - width_ * 0.5) / f_px, (v - height_ * 0.5) / f_px, 1.0};
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (double& x : d) x /= norm;
    double R[3][3];
    rotation(yaw_deg, pitch_deg, roll_deg, R);
    for (int i = 0; i < 3; ++i)
        dw[i] = R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2];
}

void RangeEstimator::locate(RangeEstimate& est, const TargetState& t,
                            const TeleFrame* tele) const {
    if (!tele || !tele->has_gps || !tele->has_heading || est.ground_m < 0.0f) return;
    const double focal = focal_mm(tele);
    if (!(focal > 0.0)) return;

    double pitch = 0.0, roll = 0.0;
    attitude(tele, pitch, roll);   // level when unknown: bearing = heading + pixel angle
    double dw[3];
    world_ray(t, focal / 36.0 * width_, tele->heading, pitch, roll, dw);
    // A ray pointing (almost) straight down has no horizontal direction; the
    // target is then under the drone and the bearing does not matter.
    const double bearing = std::hypot(dw[0], dw[1]) > 1e-6
                               ? std::atan2(dw[0], dw[1]) * kRadToDeg
                               : tele->heading;
    destination(tele->lat, tele->lon, bearing, est.ground_m, est.lat, est.lon);
    est.bearing_deg = static_cast<float>(wrap180(bearing) < 0.0 ? wrap180(bearing) + 360.0
                                                                : wrap180(bearing));
    est.has_geo = true;
}

RangeEstimate RangeEstimator::ground_plane(const TargetState& t, const TeleFrame* tele,
                                           double f_px) const {
    RangeEstimate est;

    double pitch = 0.0, roll = 0.0;
    if (!attitude(tele, pitch, roll)) return est;

    // Camera height above the target's ground; barometric rel_alt is relative
    // to the takeoff point, ground_alt_offset_m corrects for different terrain.
    if (!tele) return est;
    const double h = tele->rel_alt - cfg_.ground_alt_offset_m;
    if (h <= 0.0) return est;

    double dw[3];
    world_ray(t, f_px, 0.0, pitch, roll, dw);

    // The ray must descend meaningfully: near-horizon rays blow the range up
    // (range ~ h/tan(depression)), so treat them as no-fix rather than emit
    // a wild number. -dw[2] = sin(depression).
    if (-dw[2] < std::sin(cfg_.min_depression_deg * kDeg)) return est;

    const double slant = h / -dw[2];
    if (slant > cfg_.max_range_m) return est;
    est.slant_m = static_cast<float>(slant);
    est.ground_m = static_cast<float>(slant * std::hypot(dw[0], dw[1]));
    est.method = RangeMethod::GroundPlane;
    return est;
}

RangeEstimate RangeEstimator::raw_estimate(const TargetState& t,
                                           const TeleFrame* tele) const {
    RangeEstimate est;

    const double focal = focal_mm(tele);
    if (!(focal > 0.0)) return est;
    const double f_px = focal / 36.0 * width_;

    // Ground-plane with attitude beats the size prior when it produces a fix.
    est = ground_plane(t, tele, f_px);
    if (est.method != RangeMethod::None) return est;

    // Size prior: typical class size vs the box's longest dimension.
    const auto prior = prior_by_class_.find(t.class_id);
    if (prior != prior_by_class_.end()) {
        const float dim_px = std::max(t.box.w, t.box.h);
        if (dim_px >= 2.0f) {
            const double slant = f_px * prior->second / dim_px;
            if (slant > 0.0 && slant <= cfg_.max_range_m) {
                est.slant_m = static_cast<float>(slant);
                est.method = RangeMethod::SizePrior;
            }
        }
    }
    return est;
}

RangeEstimate RangeEstimator::update(const TargetState& target, const TeleFrame* tele) {
    if (!cfg_.enabled) return {};
    if (target.lock_id != lock_id_) {   // new lock: forget the previous window
        lock_id_ = target.lock_id;
        window_method_ = RangeMethod::None;
        slant_win_.clear();
    }
    // Locked/Coasting only — a Lost box is a stale prediction, and estimating
    // through it would smear the window with garbage.
    if (!target.valid()) return {};

    const RangeEstimate raw = raw_estimate(target, tele);
    if (raw.method == RangeMethod::None) return {};

    // Different methods disagree systematically; never median across them.
    if (raw.method != window_method_) {
        window_method_ = raw.method;
        slant_win_.clear();
    }
    slant_win_.push_back(raw.slant_m);
    const size_t win = static_cast<size_t>(std::max(1, cfg_.smooth_window));
    while (slant_win_.size() > win) slant_win_.pop_front();

    RangeEstimate est;
    est.method = raw.method;
    est.slant_m = median(slant_win_);
    est.ground_m = static_cast<float>(ground_from_slant(est.slant_m, tele));
    locate(est, target, tele);
    return est;
}

}  // namespace ot
