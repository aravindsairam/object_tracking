// Headless check: run the real YoloDetector (via the factory) on one frame and
// print detections. No GUI. Not part of the app build.
#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/video_source.hpp"

#include <chrono>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_detect <video> <config.yaml> [frame=300]\n"); return 2; }
    const int target = (argc >= 4) ? std::atoi(argv[3]) : 300;
    try {
        ot::Config cfg = ot::load_config(argv[2]);
        ot::VideoSource src(argv[1], cfg.work_height > 0 ? cfg.work_height : 640);
        auto det = ot::make_detector(cfg.detector, cfg.tiling);
        ot::ClassMap cm = ot::ClassMap::preset(cfg.detector.class_map);

        cv::Mat frame;
        while (src.frame_index() < target && src.read(frame)) {}
        if (frame.empty()) { std::fprintf(stderr, "could not reach frame %d\n", target); return 1; }
        std::printf("working frame %dx%d (idx %lld)\n", frame.cols, frame.rows,
                    (long long)src.frame_index());

        auto dets = det->detect(frame);
        std::printf("detections: %zu\n", dets.size());
        for (size_t i = 0; i < dets.size() && i < 20; ++i) {
            const auto& d = dets[i];
            std::printf("  %-14s %.2f  box[%.0f,%.0f %.0fx%.0f]\n",
                        cm.label(d.class_id).c_str(), d.score,
                        d.box.x, d.box.y, d.box.w, d.box.h);
        }
        // Timing loop (warm) on the same frame.
        for (int i = 0; i < 5; ++i) det->detect(frame);
        const int N = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) det->detect(frame);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / N;
        std::printf("detect() avg %.2f ms/frame (%.0f fps)\n", ms, 1000.0 / ms);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
