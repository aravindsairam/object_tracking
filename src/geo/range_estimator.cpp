// Locked-target distance estimation. See range_estimator.hpp for the method
// chain and docs/2d_distance/range_design.md for the derivation and error
// budget (size prior ~10-15%, dominated by how typical the class prior is).
#include "ot/range_estimator.hpp"

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

void RangeEstimator::reset() {
    lock_id_ = -2;
    window_method_ = RangeMethod::None;
    slant_win_.clear();
}

double RangeEstimator::ground_from_slant(double slant_m, const TeleFrame* tele) const {
    // Flat-ground geometry: ground = sqrt(slant^2 - height^2). Holds for both
    // methods; unknown (<0) without a usable camera height above the target.
    if (!tele) return -1.0;
    const double h = tele->rel_alt - cfg_.ground_alt_offset_m;
    if (h <= 0.0 || slant_m <= h) return -1.0;
    return std::sqrt(slant_m * slant_m - h * h);
}

RangeEstimate RangeEstimator::raw_estimate(const TargetState& t,
                                           const TeleFrame* tele) const {
    RangeEstimate est;

    const double focal = cfg_.equiv_focal_mm > 0.0f ? cfg_.equiv_focal_mm
                       : (tele && tele->has_focal) ? tele->focal_len : 0.0;
    if (!(focal > 0.0)) return est;
    const double f_px = focal / 36.0 * width_;

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
    return est;
}

}  // namespace ot
