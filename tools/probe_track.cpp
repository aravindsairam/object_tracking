// Headless check: run detect+track over a frame range and report id stability.
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/mot_tracker.hpp"
#include "ot/overlay.hpp"
#include "ot/video_source.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <map>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_track <video> <config> [start=300] [count=40]\n"); return 2; }
    const int start = (argc >= 4) ? std::atoi(argv[3]) : 300;
    const int count = (argc >= 5) ? std::atoi(argv[4]) : 40;
    try {
        ot::Config cfg = ot::load_config(argv[2]);
        ot::VideoSource src(argv[1], cfg.work_height > 0 ? cfg.work_height : 640);
        auto det = ot::make_detector(cfg.detector, cfg.tiling);
        ot::MotTracker trk(static_cast<int>(src.fps()));
        const ot::ClassMap cm = ot::ClassMap::preset(cfg.detector.class_map);

        cv::Mat f;
        while (src.frame_index() < start - 1 && src.read(f)) {}

        std::map<int, int> seen;  // id -> #frames seen
        for (int k = 0; k < count && src.read(f); ++k) {
            auto dets = det->detect(f);
            auto tr = trk.update(f, dets);
            cv::Mat view = f.clone();          // mirror the GUI: clone + draw each frame
            ot::draw_tracks(view, tr, cm);
            std::printf("frame %lld: det %2zu trk %2zu ids:",
                        (long long)src.frame_index(), dets.size(), tr.size());
            for (auto& t : tr) { std::printf(" %d", t.id); seen[t.id]++; }
            std::printf("\n");
        }
        int persistent = 0;
        for (auto& kv : seen) if (kv.second >= count / 2) ++persistent;
        std::printf("\nunique ids = %zu over %d frames; ids present in >=50%% of frames = %d\n",
                    seen.size(), count, persistent);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
