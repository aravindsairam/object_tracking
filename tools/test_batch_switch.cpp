// Regression test: one detector session must survive BATCH-SIZE SWITCHES.
// The app does this constantly — full-frame SAHI runs ~8-12 tiles, then the
// locked-ROI fast path detects a single small crop (batch 1), then a periodic
// full sweep goes wide again. A backend that caches per-shape state (e.g. an
// IoBinding output tensor materialized at one batch) must not leak it into the
// next call. Synthetic frames, no GUI, no clip needed.
//
//   ./build/test_batch_switch [config.yaml]     (default: rfdetr_medium_sahi_ocsort.yaml)
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"

#include <opencv2/core.hpp>

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    const char* cfg_path = argc > 1 ? argv[1] : "configs/rfdetr_medium_sahi_ocsort.yaml";
    ot::Config cfg = ot::load_config(cfg_path);
    auto det = ot::make_detector(cfg.detector, cfg.tiling);

    cv::RNG rng(42);
    cv::Mat full(1080, 1920, CV_8UC3), roi(300, 400, CV_8UC3), tiny(80, 100, CV_8UC3);
    rng.fill(full, cv::RNG::UNIFORM, 0, 255);
    rng.fill(roi,  cv::RNG::UNIFORM, 0, 255);
    rng.fill(tiny, cv::RNG::UNIFORM, 0, 255);

    // full (batch ~12 with overlap 0.2) -> crop (batch 1) mimics click-to-lock;
    // going back and forth catches stale state in BOTH directions.
    struct Step { const char* name; const cv::Mat* img; };
    const Step steps[] = {{"full-SAHI", &full}, {"roi-crop", &roi}, {"full-SAHI", &full},
                          {"tiny-crop", &tiny}, {"roi-crop", &roi}, {"full-SAHI", &full}};
    for (const auto& s : steps) {
        try {
            const auto dets = det->detect(*s.img);
            std::printf("  %-9s %dx%d -> %zu dets\n",
                        s.name, s.img->cols, s.img->rows, dets.size());
        } catch (const std::exception& e) {
            std::printf("FAIL at %s %dx%d: %s\n", s.name, s.img->cols, s.img->rows, e.what());
            return 1;
        }
    }
    std::printf("PASS: batch switches survive one session\n");
    return 0;
}
