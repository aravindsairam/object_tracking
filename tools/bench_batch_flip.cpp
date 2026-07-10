// Quantifies the TensorRT batch-shape-switch cost the locked-ROI path pays.
//
// While locked, the app alternates engine batch sizes: crop detects run batch-1
// every frame and the periodic full-SAHI sweep (lock.roi_full_interval) runs
// batch-8..12 — and a TensorRT execution context pays a re-activation penalty
// each time its input shape changes. This tool measures detect() latency in
// three regimes: full-frame only (steady), crop only (steady), and alternating
// (the locked-mode pattern) — the alternating-minus-steady delta IS the flip
// cost. Run it before/after a change (e.g. a dedicated batch-1 session) to
// prove the change pays for itself; if the delta is ~0, such a change is
// worthless. Synthetic frames (content doesn't matter for engine timing).
//
//   ./build/bench_batch_flip <config.yaml> [iters=60]
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

struct Stats { double avg, p50, p99; };

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto q = [&](double p) { return v[static_cast<size_t>(p * (v.size() - 1))]; };
    return {sum / v.size(), q(0.5), q(0.99)};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <config.yaml> [iters]\n", argv[0]); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 60;

    ot::Config cfg = ot::load_config(argv[1]);
    auto det = ot::make_detector(cfg.detector, cfg.tiling);

    cv::RNG rng(42);
    cv::Mat full(1080, 1920, CV_8UC3), crop(300, 400, CV_8UC3);
    rng.fill(full, cv::RNG::UNIFORM, 0, 255);
    rng.fill(crop, cv::RNG::UNIFORM, 0, 255);

    for (int i = 0; i < 10; ++i) { det->detect(full); det->detect(crop); }  // warm-up

    std::vector<double> a, b, cf, cc;
    for (int i = 0; i < iters; ++i) {          // A: full-frame steady state
        const auto t0 = std::chrono::steady_clock::now();
        det->detect(full);
        a.push_back(ms_since(t0));
    }
    for (int i = 0; i < iters; ++i) {          // B: crop steady state
        const auto t0 = std::chrono::steady_clock::now();
        det->detect(crop);
        b.push_back(ms_since(t0));
    }
    for (int i = 0; i < iters; ++i) {          // C: alternating (locked-mode pattern)
        auto t0 = std::chrono::steady_clock::now();
        det->detect(full);
        cf.push_back(ms_since(t0));
        t0 = std::chrono::steady_clock::now();
        det->detect(crop);
        cc.push_back(ms_since(t0));
    }

    const Stats sa = stats(a), sb = stats(b), scf = stats(cf), scc = stats(cc);
    std::printf("\ndetect() latency, %d iters/regime (ms):\n", iters);
    std::printf("  %-28s avg %7.2f  p50 %7.2f  p99 %7.2f\n", "full-SAHI steady", sa.avg, sa.p50, sa.p99);
    std::printf("  %-28s avg %7.2f  p50 %7.2f  p99 %7.2f\n", "crop(b1) steady", sb.avg, sb.p50, sb.p99);
    std::printf("  %-28s avg %7.2f  p50 %7.2f  p99 %7.2f\n", "full-SAHI alternating", scf.avg, scf.p50, scf.p99);
    std::printf("  %-28s avg %7.2f  p50 %7.2f  p99 %7.2f\n", "crop(b1) alternating", scc.avg, scc.p50, scc.p99);

    const double flip_full = scf.avg - sa.avg;
    const double flip_crop = scc.avg - sb.avg;
    std::printf("\nshape-flip overhead: full %+.2f ms, crop %+.2f ms  (alternating - steady)\n",
                flip_full, flip_crop);
    std::printf("locked-mode amortized (1 flip pair / %d frames): %+.3f ms/frame\n",
                20, (flip_full + flip_crop) / 20.0);
    return 0;
}
