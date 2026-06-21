// Headless check of the lock state machine: pick a real track at the start
// frame (simulating a click on its center), then follow it and print the lock
// state each frame.
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/lock_manager.hpp"
#include "ot/mot_tracker.hpp"
#include "ot/reid.hpp"
#include "ot/video_source.hpp"

#include <cstdio>
#include <cstdlib>

static const char* sname(ot::LockState s) {
    switch (s) {
        case ot::LockState::Acquiring: return "ACQUIRING";
        case ot::LockState::Locked:    return "LOCKED";
        case ot::LockState::Coasting:  return "COASTING";
        case ot::LockState::Lost:      return "LOST";
    }
    return "?";
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_lock <video> <config> [start=300] [count=60]\n"); return 2; }
    const int start = (argc >= 4) ? std::atoi(argv[3]) : 300;
    const int count = (argc >= 5) ? std::atoi(argv[4]) : 60;
    try {
        ot::Config cfg = ot::load_config(argv[2]);
        ot::VideoSource src(argv[1], cfg.work_height > 0 ? cfg.work_height : 640);
        auto det = ot::make_detector(cfg.detector, cfg.tiling);
        ot::MotTracker trk(static_cast<int>(src.fps()));
        ot::LockManager lock(src.fps(), cfg.lock.coast_to_lost,
                             cfg.lock.reacquire_thresh, ot::make_histogram_embedder());

        cv::Mat f;
        while (src.frame_index() < start - 1 && src.read(f)) {}
        if (!src.read(f)) { std::fprintf(stderr, "could not reach start frame\n"); return 1; }

        // Prime the tracker, then lock the highest-confidence track (simulated click).
        auto tracks = trk.update(f, det->detect(f));
        if (tracks.empty()) { std::fprintf(stderr, "no tracks at start frame\n"); return 1; }
        const ot::Track* pick = &tracks[0];
        for (const auto& t : tracks) if (t.score > pick->score) pick = &t;
        std::printf("locking track #%d (score %.2f) at center (%.0f,%.0f)\n",
                    pick->id, pick->score, pick->box.center().x, pick->box.center().y);
        lock.select(pick->box.center(), f, tracks);

        int locked = 0, coasting = 0, lost = 0;
        for (int k = 0; k < count && src.read(f); ++k) {
            tracks = trk.update(f, det->detect(f));
            ot::TargetState ts = lock.update(f, tracks);
            if (ts.state == ot::LockState::Locked) ++locked;
            else if (ts.state == ot::LockState::Coasting) ++coasting;
            else if (ts.state == ot::LockState::Lost) ++lost;
            if (k < 12 || ts.state != ot::LockState::Locked)
                std::printf("  f%lld %-9s box[%.0f,%.0f %.0fx%.0f] conf %.2f vel(%.0f,%.0f)\n",
                            (long long)src.frame_index(), sname(ts.state),
                            ts.box.x, ts.box.y, ts.box.w, ts.box.h, ts.confidence,
                            ts.velocity_px_s.x, ts.velocity_px_s.y);
        }
        std::printf("summary over %d frames: locked=%d coasting=%d lost=%d\n",
                    count, locked, coasting, lost);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
