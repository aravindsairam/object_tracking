// A/B throughput bench for the inference-loop software pipeline: runs the same
// frames through (a) the serial schedule detect(N) -> track(N) and (b) the
// pipelined schedule used by src/app/main.cpp — launch detect(N+1) on a worker
// before consuming N's result, so the tracker tail overlaps the next GPU run.
// Detections are identical either way (same frames, same order, same detect
// calls, serialized on one mutex); only the schedule differs. Frames are
// pre-decoded to RAM so decode never binds the measurement.
//
//   ./build/bench_pipeline <clip> <config.yaml> [frames=150]
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/mot_tracker.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s <clip> <config.yaml> [frames]\n"
                     "(no app flags here — this is a headless throughput A/B)\n", argv[0]);
        return 2;
    }
    long want_l = 150;
    if (argc == 4) {
        want_l = std::atol(argv[3]);
        if (want_l < 10) {
            std::fprintf(stderr, "frames must be a number >= 10, got '%s'\n", argv[3]);
            return 2;
        }
    }
    const size_t want = static_cast<size_t>(want_l);

    ot::Config cfg = ot::load_config(argv[2]);
    auto det = ot::make_detector(cfg.detector, cfg.tiling);

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::fprintf(stderr, "cannot open '%s'\n", argv[1]); return 1; }
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (frames.size() < want && cap.read(frame)) {
        if (cfg.work_height > 0 && frame.rows > cfg.work_height) {
            const int w = frame.cols * cfg.work_height / frame.rows;
            cv::resize(frame, frame, cv::Size(w, cfg.work_height), 0, 0, cv::INTER_AREA);
        }
        frames.push_back(frame.clone());
    }
    const size_t N = frames.size();
    if (N < 10) { std::fprintf(stderr, "decoded only %zu frames — need >= 10\n", N); return 1; }
    std::fprintf(stderr, "loaded %zu frames at %dx%d\n", N, frames[0].cols, frames[0].rows);

    for (int i = 0; i < 8; ++i) det->detect(frames[i % N]);   // warm-up

    // ---- (a) serial schedule -------------------------------------------------
    {
        ot::MotTracker tracker(cfg.tracker, 30, nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t n = 0; n < N; ++n) {
            auto dets = det->detect(frames[n]);
            tracker.update(frames[n], dets);
        }
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        std::printf("serial:    %.2f fps  (%.2f ms/frame)\n", N / s, 1000.0 * s / N);
    }

    // ---- (b) pipelined schedule (mirrors src/app/main.cpp) --------------------
    {
        ot::MotTracker tracker(cfg.tracker, 30, nullptr);
        std::mutex det_mutex;
        auto launch = [&](size_t n) {
            return std::async(std::launch::async, [&, n] {
                std::lock_guard<std::mutex> g(det_mutex);
                return det->detect(frames[n]);
            });
        };
        const auto t0 = std::chrono::steady_clock::now();
        auto pending = launch(0);
        for (size_t n = 0; n < N; ++n) {
            std::future<std::vector<ot::Detection>> next;
            if (n + 1 < N) next = launch(n + 1);   // queues behind pending on det_mutex
            auto dets = pending.get();
            tracker.update(frames[n], dets);       // overlaps detect(n+1)
            pending = std::move(next);
        }
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        std::printf("pipelined: %.2f fps  (%.2f ms/frame)\n", N / s, 1000.0 * s / N);
    }
    return 0;
}
