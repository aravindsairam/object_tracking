// Unit tests for the lock-reticle animation helpers (pure math; no GPU, model,
// or rendering). These guard the acquire-animation state machine and the
// bracket geometry that LockReticle::draw is built on.
#include "ot/overlay.hpp"
#include "ot/types.hpp"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}
bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

using ot::LockState;

// The acquire animation must fire when a target appears or recovers, but NOT on
// the frequent Coasting<->Locked detection-gap blips (else it flashes forever).
void test_acquire_transition() {
    std::printf("test: is_acquire_transition truth table\n");
    check(ot::is_acquire_transition(LockState::Acquiring, LockState::Locked),
          "Acquiring -> Locked is an acquire");
    check(ot::is_acquire_transition(LockState::Lost, LockState::Locked),
          "Lost -> Locked (recovery) is an acquire");
    check(!ot::is_acquire_transition(LockState::Coasting, LockState::Locked),
          "Coasting -> Locked is NOT an acquire");
    check(!ot::is_acquire_transition(LockState::Locked, LockState::Locked),
          "Locked -> Locked is NOT an acquire");
    check(!ot::is_acquire_transition(LockState::Acquiring, LockState::Coasting),
          "-> Coasting is NOT an acquire");
}

void test_acquire_progress() {
    std::printf("test: acquire_progress clamps and is linear in between\n");
    check(approx(ot::acquire_progress(10.0, 10.00, 0.3), 0.0f), "starts at 0");
    check(approx(ot::acquire_progress(10.0, 10.30, 0.3), 1.0f), "reaches 1 at the end");
    check(approx(ot::acquire_progress(10.0, 99.00, 0.3), 1.0f), "clamps above 1");
    check(approx(ot::acquire_progress(10.0,  9.00, 0.3), 0.0f), "clamps below 0");
    check(approx(ot::acquire_progress(10.0, 10.15, 0.3), 0.5f), "linear at the midpoint");
    check(approx(ot::acquire_progress(0.0,   5.00, 0.0), 1.0f), "zero duration -> done");
}

void test_ease_out_cubic() {
    std::printf("test: ease_out_cubic endpoints, shape, clamping\n");
    check(approx(ot::ease_out_cubic(0.0f), 0.0f), "ease(0) = 0");
    check(approx(ot::ease_out_cubic(1.0f), 1.0f), "ease(1) = 1");
    check(ot::ease_out_cubic(0.5f) > 0.5f, "ease-out sits above linear mid-way");
    check(approx(ot::ease_out_cubic(-1.0f), 0.0f), "clamps negative input");
    check(approx(ot::ease_out_cubic(2.0f), 1.0f), "clamps input above 1");
}

void test_bracket_len() {
    std::printf("test: bracket_len ~20%% of shorter side, clamped\n");
    check(ot::bracket_len({0, 0, 500, 400}) == 60, "large box clamps to max 60");   // 80 -> 60
    check(ot::bracket_len({0, 0,  10,  10}) ==  8, "tiny box clamps to min 8");      //  2 -> 8
    check(ot::bracket_len({0, 0, 200, 150}) == 30, "mid box uses ~20%% of shorter"); // 30
}

}  // namespace

int main() {
    test_acquire_transition();
    test_acquire_progress();
    test_ease_out_cubic();
    test_bracket_len();
    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
