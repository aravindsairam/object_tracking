// Unit test for the locked-ROI search window (no GPU/model/video — pure geometry).
//
// The window is cut out of the frame with `img(roi)`, which THROWS if the rect
// leaves the image — and that runs on the inference thread, so a bad rect aborts
// the app. It must also stay square: the crop is fed through the SAHI tiler, so a
// border-clipped 320x278 crop silently became two 278x278 tiles instead of one.
//
//   ./build/test_roi
#include "ot/roi.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

constexpr int W = 1920, H = 1080;
constexpr double kFps = 30.0;
constexpr float kScale = 3.0f;

ot::TargetState target(float cx, float cy, float bw, float bh, float vx = 0, float vy = 0) {
    ot::TargetState t;
    t.state = ot::LockState::Locked;
    t.center_px = {cx, cy};
    t.box = {cx - bw / 2, cy - bh / 2, bw, bh};
    t.velocity_px_s = {vx, vy};
    return t;
}

bool inside(const cv::Rect& r, int w = W, int h = H) {
    return r.width > 0 && r.height > 0 && r.x >= 0 && r.y >= 0 &&
           r.x + r.width <= w && r.y + r.height <= h;
}
bool square(const cv::Rect& r) { return r.width == r.height; }
bool contains(const cv::Rect& r, float x, float y) {
    return x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height;
}

void test_interior_is_centred_square() {
    std::printf("test: interior target -> centred 320 square\n");
    const cv::Rect r = ot::predict_roi(target(960, 540, 60, 40), kScale, kFps, W, H);
    check(inside(r) && square(r), "inside the frame and square");
    check(r.width == 320, "180 px window floored to 320");
    check(std::abs(r.x + r.width / 2 - 960) <= 1 && std::abs(r.y + r.height / 2 - 540) <= 1,
          "centred on the target");
}

void test_near_edges_stay_square() {
    std::printf("test: target near an edge -> full square slid inside, target still in it\n");
    const struct { float x, y; const char* name; } cases[] = {
        {1900, 540, "right edge"},  {10, 540, "left edge"},
        {960, 1075, "bottom edge"}, {960, 5, "top edge"},
        {1915, 1075, "bottom-right corner"}, {3, 2, "top-left corner"},
    };
    for (const auto& c : cases) {
        const cv::Rect r = ot::predict_roi(target(c.x, c.y, 60, 40), kScale, kFps, W, H);
        char msg[96];
        std::snprintf(msg, sizeof msg, "%s: inside, 320x320, contains target", c.name);
        check(inside(r) && square(r) && r.width == 320 && contains(r, c.x, c.y), msg);
    }
}

void test_prediction_off_frame_does_not_leave_image() {
    std::printf("test: predicted centre far off-frame -> rect still inside (was a crash)\n");
    // One frame of velocity carries the predicted centre well past each edge.
    const struct { float vx, vy; const char* name; } cases[] = {
        {+30000, 0, "past right"},  {-30000, 0, "past left"},
        {0, +30000, "past bottom"}, {0, -30000, "past top"},
        {+1e9f, +1e9f, "absurd velocity"},
    };
    for (const auto& c : cases) {
        const cv::Rect r =
            ot::predict_roi(target(1800, 1000, 60, 40, c.vx, c.vy), kScale, kFps, W, H);
        char msg[96];
        std::snprintf(msg, sizeof msg, "%s: inside and square", c.name);
        check(inside(r) && square(r), msg);
    }
}

void test_non_finite_inputs() {
    std::printf("test: NaN/inf velocity or box -> rect still inside\n");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check(inside(ot::predict_roi(target(960, 540, 60, 40, nan, 0), kScale, kFps, W, H)),
          "NaN velocity");
    check(inside(ot::predict_roi(target(960, 540, 60, 40, inf, -inf), kScale, kFps, W, H)),
          "inf velocity");
    check(inside(ot::predict_roi(target(960, 540, nan, nan), kScale, kFps, W, H)), "NaN box size");
    check(inside(ot::predict_roi(target(nan, nan, 60, 40), kScale, kFps, W, H)), "NaN centre");
}

void test_large_target_and_small_frame() {
    std::printf("test: window larger than the frame -> capped to the short side\n");
    const cv::Rect big = ot::predict_roi(target(1700, 900, 500, 450), kScale, kFps, W, H);
    check(inside(big) && square(big) && big.width == H, "1500 px window capped to 1080 square");
    const cv::Rect tiny = ot::predict_roi(target(100, 70, 20, 20), kScale, kFps, 200, 150);
    check(inside(tiny, 200, 150) && square(tiny) && tiny.width == 150,
          "200x150 frame -> 150 square");
}

void test_sweep_always_inside_and_square() {
    std::printf("test: sweep of centres in and around the frame\n");
    int bad = 0;
    for (float y = -2000; y <= H + 2000; y += 37)
        for (float x = -2000; x <= W + 2000; x += 41) {
            const cv::Rect r = ot::predict_roi(target(x, y, 80, 50), kScale, kFps, W, H);
            if (!inside(r) || !square(r)) ++bad;
        }
    if (bad) std::printf("    %d bad rects\n", bad);
    check(bad == 0, "every rect inside and square");
}

}  // namespace

int main() {
    test_interior_is_centred_square();
    test_near_edges_stay_square();
    test_prediction_off_frame_does_not_leave_image();
    test_non_finite_inputs();
    test_large_target_and_small_frame();
    test_sweep_always_inside_and_square();
    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
