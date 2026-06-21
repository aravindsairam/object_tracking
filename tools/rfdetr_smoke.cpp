// Headless smoke / parity check for a detector config: builds the detector via
// the real factory path, runs detect() on one frame, and prints the kept
// detections. No GUI — usable on servers. Mirrors what the Python parity check
// does, but through the actual C++ decode.
//
//   ./build/rfdetr_smoke <video> <config.yaml> [frame_index]
#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <video> <config.yaml> [frame_index]\n", argv[0]);
        return 2;
    }
    const ot::Config cfg = ot::load_config(argv[2]);
    auto det = ot::make_detector(cfg.detector, cfg.tiling);
    const ot::ClassMap cm = ot::ClassMap::preset(cfg.detector.class_map);

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "cannot open video '%s'\n", argv[1]);
        return 1;
    }
    const int frame_no = (argc > 3) ? std::atoi(argv[3]) : 300;
    cap.set(cv::CAP_PROP_POS_FRAMES, frame_no);
    cv::Mat frame;
    if (!cap.read(frame)) {
        std::fprintf(stderr, "cannot read frame %d\n", frame_no);
        return 1;
    }
    // Match the app: downscale to the config's working height (input.height) so
    // SAHI tile counts and timings reflect what object_tracking actually runs.
    if (cfg.work_height > 0 && frame.rows > cfg.work_height) {
        const int w = frame.cols * cfg.work_height / frame.rows;
        cv::resize(frame, frame, cv::Size(w, cfg.work_height), 0, 0, cv::INTER_AREA);
    }

    det->detect(frame);  // warm up (model load + first-run CUDA graph)
    const auto t0 = std::chrono::steady_clock::now();
    const auto dets = det->detect(frame);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("frame %dx%d (idx %d): %zu detections [class_map=%s]  %.1f ms (%.1f fps)\n",
                frame.cols, frame.rows, frame_no, dets.size(), cfg.detector.class_map.c_str(),
                ms, ms > 0 ? 1000.0 / ms : 0.0);
    int n = 0;
    for (const auto& d : dets) {
        if (n++ >= 20) break;
        std::printf("  %-12s %.2f  box=(%.0f,%.0f,%.0f,%.0f)\n",
                    cm.label(d.class_id).c_str(), d.score,
                    d.box.x, d.box.y, d.box.w, d.box.h);
    }
    return 0;
}
