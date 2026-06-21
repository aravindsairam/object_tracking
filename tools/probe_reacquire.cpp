// Headless test of appearance re-acquisition: lock a target, simulate an
// occlusion (drop its id), then have it reappear with a NEW id and verify the
// LockManager re-binds by appearance+motion (not id continuity).
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/lock_manager.hpp"
#include "ot/mot_tracker.hpp"
#include "ot/reid.hpp"
#include "ot/video_source.hpp"

#include <algorithm>
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
    if (argc < 3) { std::fprintf(stderr, "usage: probe_reacquire <video> <config> [start=300] [gap=5]\n"); return 2; }
    const int start = (argc >= 4) ? std::atoi(argv[3]) : 300;
    const int gap   = (argc >= 5) ? std::atoi(argv[4]) : 5;
    try {
        ot::Config cfg = ot::load_config(argv[2]);
        ot::VideoSource src(argv[1], cfg.work_height > 0 ? cfg.work_height : 640);
        auto det = ot::make_detector(cfg.detector, cfg.tiling);
        ot::MotTracker trk(static_cast<int>(src.fps()));
        ot::LockManager lock(src.fps(), cfg.lock.coast_to_lost,
                             cfg.lock.reacquire_thresh, ot::make_histogram_embedder());

        cv::Mat f;
        while (src.frame_index() < start - 1 && src.read(f)) {}
        if (!src.read(f)) return 1;
        auto tracks = trk.update(f, det->detect(f));
        if (tracks.empty()) { std::fprintf(stderr, "no tracks\n"); return 1; }
        const ot::Track* pick = &tracks[0];
        for (const auto& t : tracks) if (t.score > pick->score) pick = &t;
        const int L = pick->id;
        const int NEWID = L + 5000;
        std::printf("locked id %d; will occlude for %d frames then reappear as id %d\n", L, gap, NEWID);
        lock.select(pick->box.center(), f, tracks);

        for (int k = 0; k < 3 + gap + 15 && src.read(f); ++k) {
            tracks = trk.update(f, det->detect(f));
            std::vector<ot::Track> fed = tracks;
            const char* phase;
            if (k < 3) {
                phase = "normal";
            } else if (k < 3 + gap) {                       // occlusion: drop the id
                fed.erase(std::remove_if(fed.begin(), fed.end(),
                          [&](const ot::Track& t){ return t.id == L; }), fed.end());
                phase = "occluded";
            } else {                                         // reappear with a NEW id
                for (auto& t : fed) if (t.id == L) t.id = NEWID;
                phase = "reappeared(newid)";
            }
            ot::TargetState ts = lock.update(f, fed);
            std::printf("  k%2d %-17s -> %-9s locked_id=%d conf %.2f\n",
                        k, phase, sname(ts.state), lock.locked_id(), ts.confidence);
        }
        std::printf("final locked_id = %d (expected %d if re-acquired)\n", lock.locked_id(), NEWID);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
