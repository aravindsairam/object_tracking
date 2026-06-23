#pragma once

#include <cmath>

namespace ot {

// One-Euro filter (Casiez et al., 2012): a speed-adaptive low-pass for a noisy
// signal. Smooths hard when the signal is slow (kills jitter) and opens up when
// it moves fast (low lag) — exactly the trade-off you want for a tracked target
// you intend to aim at. Scalar; compose one per channel.
//
// Tuning: min_cutoff (Hz) sets the floor smoothing (lower = steadier but laggier);
// beta sets how aggressively the cutoff rises with speed (higher = less lag on
// fast motion). Both are applied per call so a single config knob can drive them.
class OneEuroFilter {
public:
    void reset() { has_prev_ = false; }

    // Filter `x` sampled at absolute time `t_s` (seconds). dt is derived from the
    // previous call, so variable frame intervals (dropped frames) are handled.
    float filter(float x, double t_s, float min_cutoff, float beta, float dcutoff = 1.0f) {
        if (!has_prev_) {
            x_prev_ = x;
            dx_prev_ = 0.0f;
            t_prev_ = t_s;
            has_prev_ = true;
            return x;
        }
        float dt = static_cast<float>(t_s - t_prev_);
        if (dt <= 0.0f) dt = 1e-3f;   // guard against equal/backwards timestamps
        t_prev_ = t_s;

        const float dx = (x - x_prev_) / dt;
        const float a_d = alpha(dt, dcutoff);
        const float dx_hat = a_d * dx + (1.0f - a_d) * dx_prev_;

        const float cutoff = min_cutoff + beta * std::fabs(dx_hat);
        const float a = alpha(dt, cutoff);
        const float x_hat = a * x + (1.0f - a) * x_prev_;

        x_prev_ = x_hat;
        dx_prev_ = dx_hat;
        return x_hat;
    }

private:
    static float alpha(float dt, float cutoff) {
        const float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    bool   has_prev_ = false;
    float  x_prev_ = 0.0f;
    float  dx_prev_ = 0.0f;
    double t_prev_ = 0.0;
};

}  // namespace ot
