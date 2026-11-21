#include "ot/bounded_queue.hpp"
#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/lock_manager.hpp"
#include "ot/mot_tracker.hpp"
#include "ot/one_euro.hpp"
#include "ot/overlay.hpp"
#include "ot/range_estimator.hpp"
#include "ot/reid.hpp"
#include "ot/roi.hpp"
#include "ot/selector.hpp"
#include "ot/target_sink.hpp"
#include "ot/telemetry.hpp"
#include "ot/video_source.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr char kWindow[] = "object_tracking";

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

double dur_ms(std::chrono::steady_clock::time_point a,
              std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// One-Euro smoothing of the locked box for DISPLAY/OUTPUT only. Driven by a single
// `smoothing` knob in [0,1]: 0 disables (caller skips apply), higher -> steadier but
// laggier. Applied to a copy of the target so the raw box still feeds ROI prediction.
struct BoxSmoother {
    ot::OneEuroFilter fx, fy, fw, fh;
    void reset() { fx.reset(); fy.reset(); fw.reset(); fh.reset(); }
    void apply(ot::TargetState& t, double t_s, float smoothing) {
        const float s = std::min(1.0f, std::max(0.0f, smoothing));
        const float min_cutoff = 8.0f - 7.5f * s;   // s=0 -> 8 Hz (light); s=1 -> 0.5 Hz (steady)
        const float beta = 0.02f;                    // speed adaptivity (less lag when moving fast)
        float cx = t.box.x + t.box.w * 0.5f;
        float cy = t.box.y + t.box.h * 0.5f;
        cx = fx.filter(cx, t_s, min_cutoff, beta);
        cy = fy.filter(cy, t_s, min_cutoff, beta);
        const float w = fw.filter(t.box.w, t_s, min_cutoff, beta);
        const float h = fh.filter(t.box.h, t_s, min_cutoff, beta);
        t.box = {cx - w * 0.5f, cy - h * 0.5f, w, h};
        t.center_px = {cx, cy};
    }
};

// A decoded working-resolution frame. Used both for the capture->inference tap
// (newest-wins, via push_latest) and the capture->display stream (every frame).
struct CapturedFrame {
    int64_t index = 0;
    cv::Mat img;
};

// The inference thread's latest output, latched for the display thread to read.
//
// The display runs at a steady video cadence and inference runs at its own
// (variable) rate, so the display reads whatever the MOST RECENT detection result
// is and paints it onto the live frame. This is what decouples smooth playback
// from inference jitter: the video never waits for a detection. A slow inference
// frame just means the overlay is one tick stale, which LockManager's velocity
// prediction already smooths over.
struct SharedResult {
    std::mutex             m;
    std::vector<ot::Track> tracks;
    ot::TargetState        target;
    bool                   locked = false;
    bool                   roi = false;     // detection ran on the locked ROI fast-path
    bool                   valid = false;   // false until the first detection lands
    int64_t                src_index = -1;  // frame index this result was computed from
    double                 infer_ms = 0.0;  // detect+track stage time
    double                 det_fps = 0.0;   // inference loop rate (EMA)

    void set(std::vector<ot::Track> t, const ot::TargetState& tg, bool lk, bool roi_,
             int64_t idx, double ims, double dfps) {
        std::lock_guard<std::mutex> g(m);
        tracks = std::move(t);
        target = tg;
        locked = lk;
        roi = roi_;
        valid = true;
        src_index = idx;
        infer_ms = ims;
        det_fps = dfps;
    }

    // Snapshot the latest result for one display frame. Returns false until the
    // first detection has landed (so the very first frames show clean video).
    bool get(std::vector<ot::Track>& t, ot::TargetState& tg, bool& lk, bool& roi_,
             int64_t& idx, double& ims, double& dfps) {
        std::lock_guard<std::mutex> g(m);
        if (!valid) return false;
        t = tracks;
        tg = target;
        lk = locked;
        roi_ = roi;
        idx = src_index;
        ims = infer_ms;
        dfps = det_fps;
        return true;
    }
};

// UI -> inference command channel. The window/mouse/keys live on the main
// (display) thread, but the lock state lives on the inference thread, so clicks
// and resets are posted here and applied at the top of the inference loop.
struct UiCommands {
    std::mutex   m;
    bool         reset = false;
    bool         has_click = false;
    cv::Point2f  click{0, 0};

    void post_click(cv::Point2f p) {
        std::lock_guard<std::mutex> lk(m);
        has_click = true;
        click = p;
    }
    void post_reset() {
        std::lock_guard<std::mutex> lk(m);
        reset = true;
    }
    void drain(bool& do_reset, bool& do_click, cv::Point2f& c) {
        std::lock_guard<std::mutex> lk(m);
        do_reset = reset;
        do_click = has_click;
        c = click;
        reset = false;
        has_click = false;
    }
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <video|rtsp|camera-index> [config.yaml] [options]\n"
            "  --height N     working resolution (downscale to N px tall; 0=native)\n"
            "  --start N      seek to frame N before starting\n"
            "  --record FILE  write the annotated video to FILE (e.g. out.mp4)\n"
            "  --loop         restart at end of stream\n"
            "  --profile      print pipeline timing to stderr every 120 frames\n"
            "  --autolock     auto-lock the most central track (centered tracking / headless)\n"
            "  --out FILE     write per-frame locked-target JSONL to FILE (forces jsonl sink;\n"
            "                 for A/B eval: same clip + --autolock, one FILE per tracker config)\n"
            "  --display-fps N  cap display to N fps (default: source fps, max 60; 0=uncapped)\n"
            "  --zoom         show a magnified side panel of the locked target\n"
            "  --zoom-width N width of the zoom side panel in px (default: 360)\n"
            "  --telemetry FILE  drone telemetry file (default: auto-detect next to the video)\n"
            "  --gimbal-pitch DEG  fixed gimbal pitch for ground-plane ranging (0=level, -90=nadir;\n"
            "                 overrides telemetry attitude; for clips with a known, held pitch)\n"
            "  keys:  click=lock  r=reset  space=pause  s=screenshot  q/ESC=quit\n",
            argv[0]);
        return 2;
    }

    std::string config_path = "configs/waldo.yaml";
    std::string record_path;
    int  target_height = -1;       // -1 = use config
    int64_t start_frame = 0;
    bool loop = false;
    bool profile = false;
    bool autolock = false;
    std::string out_override;      // --out FILE: force jsonl sink to this path (A/B eval)
    double display_fps = -1.0;     // -1 = match source fps (capped at 60); 0 = uncapped (benchmark)
    bool zoom = false;             // show a magnified side panel of the locked target
    int  zoom_w = 360;             // zoom side-panel width in px
    std::string tele_override;      // --telemetry FILE (else auto-detect)
    bool   have_gimbal_pitch = false;
    double gimbal_pitch = 0.0;     // --gimbal-pitch DEG: fixed attitude for ranging
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--height" && i + 1 < argc)      target_height = std::stoi(argv[++i]);
        else if (a == "--start" && i + 1 < argc)  start_frame = std::stoll(argv[++i]);
        else if (a == "--record" && i + 1 < argc) record_path = argv[++i];
        else if (a == "--loop")                   loop = true;
        else if (a == "--profile")                profile = true;
        else if (a == "--autolock")               autolock = true;
        else if (a == "--out" && i + 1 < argc)    out_override = argv[++i];
        else if (a == "--display-fps" && i + 1 < argc) display_fps = std::stod(argv[++i]);
        else if (a == "--zoom")                   zoom = true;
        else if (a == "--zoom-width" && i + 1 < argc) zoom_w = std::max(120, std::stoi(argv[++i]));
        else if (a == "--telemetry" && i + 1 < argc) tele_override = argv[++i];
        else if (a == "--gimbal-pitch" && i + 1 < argc) {
            gimbal_pitch = std::stod(argv[++i]);
            have_gimbal_pitch = true;
        }
        else if (ends_with(a, ".yaml") || ends_with(a, ".yml")) config_path = a;
    }
    // Enable the backend's inference timer (read in the OrtBackend ctor) before
    // any detector is built, so --profile also reports pure-inference ms.
    if (profile) setenv("OT_PROFILE", "1", 1);

    try {
        const ot::Config cfg = ot::load_config(config_path);
        if (target_height < 0) target_height = cfg.work_height;

        ot::VideoSource source(argv[1], target_height, cfg.decoder);
        const int W = source.width(), H = source.height();
        const double fps = source.fps();
        std::printf("[run] source native %dx%d @ %.2f fps -> working %dx%d\n",
                    source.native_width(), source.native_height(), fps, W, H);
        std::printf("[run] detector: family=%s model=%s class_map=%s  tiling=%s\n",
                    cfg.detector.family.c_str(), cfg.detector.model_path.c_str(),
                    cfg.detector.class_map.c_str(), cfg.tiling.enabled ? "on" : "off");
        if (start_frame > 0) source.seek(start_frame);

        // Drone telemetry file (GPS / altitude / per-frame focal; gimbal attitude
        // when the drone records it). Auto-detected next to a file
        // source, --telemetry overrides. Absence just leaves telemetry features off.
        std::unique_ptr<ot::Telemetry> telemetry;
        const std::string tele_path = !tele_override.empty() ? tele_override
                                                             : ot::find_telemetry_sidecar(argv[1]);
        if (!tele_path.empty()) {
            try {
                telemetry = std::make_unique<ot::Telemetry>(ot::parse_telemetry(tele_path));
                std::printf("[run] telemetry: %s (%zu records%s)\n", tele_path.c_str(),
                            telemetry->size(),
                            telemetry->has_any_attitude() ? ", gimbal attitude" : "");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[run] telemetry unavailable: %s\n", e.what());
            }
        }

        auto detector = ot::make_detector(cfg.detector, cfg.tiling);
        const ot::ClassMap class_map = ot::ClassMap::preset(cfg.detector.class_map);
        // One appearance embedder, shared by the tracker (BoT-SORT association) and
        // the lock (verify + re-acquire). Created before the tracker so it can be
        // handed in; with ByteTrack only the lock uses it.
        auto reid = cfg.reid.kind == "onnx"
            ? ot::make_onnx_reid_embedder(cfg.reid.model_path, cfg.reid.input_w, cfg.reid.input_h,
                                          cfg.reid.backend, cfg.reid.device, cfg.reid.precision)
            : ot::make_histogram_embedder();
        std::printf("[run] reid: %s\n", cfg.reid.kind.c_str());
        // Warm up the ReID embedder so its slow FIRST inference happens HERE, during
        // startup, instead of inside the user's first lock click. The ONNX backend
        // defers heavy init (CUDA cuDNN autotune ~0.5 s; TensorRT engine build minutes)
        // to the first run(). With a motion-only tracker (OC-SORT / ByteTrack, or
        // BoT-SORT with_reid=false) the tracker never calls embed(), so without this
        // the entire penalty lands on the first select() — the first lock visibly lags
        // and looks like it failed to take. One throwaway embed primes the session.
        if (reid) {
            const auto tw0 = std::chrono::steady_clock::now();
            cv::Mat warm(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
            (void)reid->embed(warm, ot::BBox{16, 16, 160, 200});
            const double warm_ms = dur_ms(tw0, std::chrono::steady_clock::now());
            if (warm_ms > 20.0) std::printf("[run] reid warm-up: %.0f ms\n", warm_ms);
        }

        ot::MotTracker tracker(cfg.tracker, static_cast<int>(fps), reid);
        // Print the association metric + gate so an A/B run shows what is really
        // active (biou also shows its buffer ratio).
        char asso[64];
        if (cfg.tracker.asso_func == "biou")
            std::snprintf(asso, sizeof asso, "asso=biou r=%.2f gate=%.2f",
                          cfg.tracker.buffer_ratio, cfg.tracker.iou_threshold);
        else
            std::snprintf(asso, sizeof asso, "asso=%s gate=%.2f",
                          cfg.tracker.asso_func.c_str(), cfg.tracker.iou_threshold);
        if (cfg.tracker.type == "botsort")
            std::printf("[run] tracker: botsort %s (reid=%s, gmc=%s)\n", asso,
                        (cfg.tracker.with_reid && reid) ? "on" : "off",
                        cfg.tracker.cmc_method == "ecc" ? "ecc" : "off");
        else
            std::printf("[run] tracker: %s %s\n", cfg.tracker.type.c_str(), asso);

        // coast_to_lost is referenced to 30 fps; scale to source fps so the lock
        // tolerates the same ~wall-clock occlusion at 60 fps as it would at 30.
        const double fps_scale = fps > 0 ? fps / 30.0 : 1.0;
        const int coast = std::max(1, static_cast<int>(cfg.lock.coast_to_lost * fps_scale + 0.5));
        ot::LockManager lock(fps, coast, cfg.lock.reacquire_thresh, reid,
                             cfg.lock.verify_thresh, cfg.lock.reacquire_max_frac,
                             cfg.lock.reacquire_margin, cfg.lock.reacquire_thresh_single);
        if (cfg.lock.smoothing > 0.0f)
            std::printf("[run] output smoothing: %.2f\n", cfg.lock.smoothing);
        // Distance to the locked target, from telemetry + class-size priors.
        // Lives on the inference thread only (constructed here, used in its tail).
        ot::RangeEstimator ranger(cfg.range, W, H, class_map);
        if (have_gimbal_pitch) {
            ranger.set_static_pitch(static_cast<float>(gimbal_pitch));
            std::printf("[run] range: fixed gimbal pitch %.1f deg (ground-plane method)\n",
                        gimbal_pitch);
        }

        // --out FILE forces a jsonl sink to that path (overrides config), so each
        // tracker config can dump to its own file for an objective A/B comparison.
        auto sink = out_override.empty()
            ? ot::make_sink(cfg.output.sink, cfg.output.path)
            : ot::make_sink("jsonl", out_override);
        if (!out_override.empty())
            std::printf("[run] locked-target log -> %s\n", out_override.c_str());

        std::unique_ptr<cv::VideoWriter> recorder;
        if (!record_path.empty()) {
            recorder = std::make_unique<cv::VideoWriter>(
                record_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(W, H));
            if (!recorder->isOpened())
                throw std::runtime_error("cannot open recorder '" + record_path + "'");
            std::printf("[run] recording annotated video to %s\n", record_path.c_str());
        }

        // When the zoom panel is on, the window widens by zoom_w. The displayed
        // image (and thus the selector's click space) is the full composite size.
        const int disp_w = W + (zoom ? zoom_w : 0);
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindow, disp_w, H);
        ot::Selector selector(kWindow, cv::Size(disp_w, H));
        if (zoom) std::printf("[run] zoom side-panel enabled (%dpx wide)\n", zoom_w);
        std::printf("[run] click a target to lock.  keys: r=reset space=pause s=shot q=quit\n");

        // ---- pipeline state -------------------------------------------------
        // The capture thread feeds TWO consumers: the display gets EVERY decoded
        // frame (blocking push -> backpressure paces capture to the display rate),
        // while inference gets only the NEWEST frame (push_latest drops stale ones,
        // so a slow detection never backs up decoding). Inference publishes its
        // result to `result`, which the display latches onto the live video.
        ot::BoundedQueue<CapturedFrame> infer_q(1);   // capture -> inference (newest-wins)
        ot::BoundedQueue<CapturedFrame> disp_q(3);    // capture -> display  (every frame)
        SharedResult       result;
        UiCommands         cmds;
        std::atomic<bool>  running{true};
        std::atomic<bool>  paused{false};
        std::atomic<double> cap_ms{0.0};   // capture(decode) stage EMA, for --profile

        // ---- capture thread: decode -> {infer_q (newest), disp_q (all)} ----
        std::thread capture([&] {
            double ema = 0.0;
            cv::Mat frame;
            while (running.load()) {
                if (paused.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = source.read(frame);
                const double rms = dur_ms(t0, std::chrono::steady_clock::now());
                ema = ema == 0.0 ? rms : 0.9 * ema + 0.1 * rms;
                cap_ms.store(ema);
                if (!ok) {                                  // end of stream
                    if (loop && source.seek(0)) { cmds.post_reset(); continue; }
                    break;
                }
                const int64_t idx = source.frame_index();
                // Hand the freshest frame to inference (read-only share, newest-wins)...
                infer_q.push_latest({idx, frame});
                // ...and an independent copy to the display path (it draws overlays
                // on it). The clone keeps inference's frame pristine while display
                // mutates its own; a 1080p copy is ~0.5 ms, well inside capture's budget.
                if (!disp_q.push({idx, frame.clone()})) break;   // closed -> quitting
                frame = cv::Mat();                          // release; the infer slot owns the buffer
            }
            infer_q.close();
            disp_q.close();
        });

        // ---- inference thread: infer_q -> detect/track/lock -> result ------
        std::thread infer([&] {
            CapturedFrame in;
            ot::TargetState last_target;     // previous frame's lock state, drives the ROI crop
            int64_t infer_frame = 0;
            BoxSmoother box_smoother;        // smooths the displayed/logged box (output only)
            const int full_every = std::max(1, cfg.lock.roi_full_interval);
            double det_ema = 0.0;
            auto t_prev = std::chrono::steady_clock::now();

            // One-deep software pipeline for the FULL-SAHI path: launch frame
            // N+1's detect on a worker before blocking on frame N's result, so
            // this thread's tail work (tracker/lock/publish of N) overlaps
            // N+1's GPU time. detect() itself is not thread-safe (shared
            // staging buffers), so worker calls serialize on det_mutex — the
            // GPU is serial anyway, the overlap that pays is tail-vs-Run.
            // The locked-ROI path stays SERIAL: its crop is predicted from the
            // PREVIOUS frame's lock state, so launching ahead would predict
            // from stale state. Detections are identical to the serial loop
            // (same frames, same order, same detect calls).
            struct Inflight {
                CapturedFrame in;
                std::future<std::vector<ot::Detection>> dets;
                std::chrono::steady_clock::time_point t0;
            };
            std::optional<Inflight> ahead;
            std::mutex det_mutex;
            auto launch_full = [&](CapturedFrame&& f) {
                Inflight fl;
                fl.in = std::move(f);
                fl.t0 = std::chrono::steady_clock::now();
                const cv::Mat img = fl.in.img;   // shared header; worker reads only
                ot::Detector* det = detector.get();
                fl.dets = std::async(std::launch::async, [det, img, &det_mutex] {
                    std::lock_guard<std::mutex> g(det_mutex);
                    return det->detect(img);
                });
                return fl;
            };

            while (running.load()) {
                std::vector<ot::Detection> dets;
                bool use_roi = false;
                std::chrono::steady_clock::time_point t0;

                if (ahead) {
                    // Consume the prefetched full-SAHI frame; top up the pipeline
                    // BEFORE blocking on its result so the next detect queues
                    // behind it on the GPU while we run this frame's tail below.
                    Inflight cur = std::move(*ahead);
                    ahead.reset();
                    in = std::move(cur.in);
                    t0 = cur.t0;
                    CapturedFrame next;
                    if (last_target.state != ot::LockState::Locked &&
                        infer_q.pop_for(next, std::chrono::milliseconds(0)) == 1) {
                        ahead = launch_full(std::move(next));
                    }
                    dets = cur.dets.get();
                } else {
                    if (!infer_q.pop(in)) break;            // closed + drained

                    // Locked-ROI fast-path: while solidly locked, detect only a
                    // crop around the predicted target (cheap) — except every
                    // Nth frame, when full SAHI runs to catch new objects and
                    // correct drift. Any non-Locked state (Coasting/Lost/
                    // Acquiring) also takes the full path, so a target that
                    // slips out of the crop self-recovers.
                    use_roi = cfg.lock.roi_fastpath
                           && last_target.state == ot::LockState::Locked
                           && (infer_frame % full_every != 0);

                    t0 = std::chrono::steady_clock::now();
                    if (use_roi) {
                        const cv::Rect roi = ot::predict_roi(last_target, cfg.lock.roi_scale, fps, W, H);
                        std::lock_guard<std::mutex> g(det_mutex);
                        auto local = detector->detect(in.img(roi));
                        dets.reserve(local.size());
                        for (ot::Detection d : local) {     // shift crop-space boxes to frame space
                            d.box.x += roi.x;
                            d.box.y += roi.y;
                            dets.push_back(d);
                        }
                    } else {
                        // Enter the pipeline: launch this frame and, unless we are
                        // locked (the next frame then belongs to the ROI path),
                        // prefetch the next one behind it.
                        Inflight cur = launch_full(std::move(in));
                        in = std::move(cur.in);
                        t0 = cur.t0;
                        CapturedFrame next;
                        if (last_target.state != ot::LockState::Locked &&
                            infer_q.pop_for(next, std::chrono::milliseconds(0)) == 1) {
                            ahead = launch_full(std::move(next));
                        }
                        dets = cur.dets.get();
                    }
                }

                bool do_reset = false, do_click = false;
                cv::Point2f click;
                cmds.drain(do_reset, do_click, click);
                if (do_reset) { lock.reset(); last_target = ot::TargetState{}; ranger.reset(); }

                auto tracks = tracker.update(in.img, dets);
                const double infer_ms = dur_ms(t0, std::chrono::steady_clock::now());

                if (do_click) {
                    lock.select(click, in.img, tracks);
                } else if (autolock && lock.state() == ot::LockState::Acquiring && !tracks.empty()) {
                    // Lock the largest track: click at its center so select() binds it.
                    const ot::Track* big = &tracks[0];
                    for (const auto& t : tracks)
                        if (t.box.area() > big->box.area()) big = &t;
                    lock.select(big->box.center(), in.img, tracks);
                }
                ot::TargetState target = lock.update(in.img, tracks);
                last_target = target;        // RAW copy: feeds ROI prediction, stays responsive
                ++infer_frame;

                // Output smoothing: steady the box we DISPLAY and LOG (not the raw
                // last_target above, so ROI prediction keeps full responsiveness).
                // Reset between locks so a fresh target doesn't inherit stale state.
                if (cfg.lock.smoothing > 0.0f && target.valid())
                    box_smoother.apply(target, fps > 0 ? in.index / fps : infer_frame / 30.0,
                                       cfg.lock.smoothing);
                else
                    box_smoother.reset();

                // Distance to the locked target (smoothed box + this frame's
                // telemetry). Serial-tail safe: pure CPU trig, never touches the
                // detector or det_mutex, well under the locked-ROI time budget.
                {
                    const ot::TeleFrame* tf = telemetry ? telemetry->at(in.index) : nullptr;
                    const ot::RangeEstimate est = ranger.update(target, tf);
                    target.range_slant_m  = est.slant_m;
                    target.range_ground_m = est.ground_m;
                    target.range_method   = est.method;
                }

                const bool locked = lock.has_target() || lock.state() == ot::LockState::Lost;
                if (locked) {
                    target.frame_index = in.index;
                    target.timestamp_s = fps > 0 ? in.index / fps : 0.0;
                    target.center_norm = {target.center_px.x / W, target.center_px.y / H};
                    sink->write(target);
                }

                const auto now = std::chrono::steady_clock::now();
                const double dms = dur_ms(t_prev, now);
                t_prev = now;
                det_ema = det_ema == 0.0 ? dms : 0.9 * det_ema + 0.1 * dms;

                result.set(std::move(tracks), target, locked, use_roi, in.index, infer_ms,
                           det_ema > 0 ? 1000.0 / det_ema : 0.0);
            }
        });

        // ---- display + UI: main thread (HighGUI must run here) -------------
        // Pace the VIDEO to a steady cadence (default: source fps, capped at 60)
        // and paint the latest available detection on each frame. Because the
        // video no longer waits for inference, playback stays smooth regardless of
        // the 5ms ROI / 22ms full-SAHI swing underneath. --display-fps 0 uncaps.
        const double eff_fps = display_fps < 0.0 ? std::min(60.0, fps) : display_fps;
        const double interval_ms = eff_fps > 0.0 ? 1000.0 / eff_fps : 0.0;
        if (interval_ms > 0.0)
            std::printf("[run] display paced to %.1f fps (use --display-fps 0 to uncap)\n", eff_fps);

        CapturedFrame shown;             // last composed frame (re-shown while paused)
        bool have_shown = false;
        ot::LockReticle reticle;         // animated lock overlay (owns its acquire anim)
        double ema_disp_ms = 0.0, ema_ms = 0.0, ema_draw = 0.0, ema_show = 0.0;
        double last_det_fps = 0.0;
        const auto t_epoch = std::chrono::steady_clock::now();   // monotonic clock for animations
        auto last_show = std::chrono::steady_clock::now();
        auto last_present = std::chrono::steady_clock::now();
        long prof_n = 0;

        while (running.load()) {
            CapturedFrame df;
            const int got = disp_q.pop_for(df, std::chrono::milliseconds(30));
            if (got == -1) break;                           // pipeline drained (EOF)

            const bool fresh = (got == 1);
            if (fresh) {
                const auto now = std::chrono::steady_clock::now();
                const double now_s = dur_ms(t_epoch, now) / 1000.0;   // seconds, for animations
                const double dms = dur_ms(last_show, now);
                last_show = now;
                ema_disp_ms = ema_disp_ms == 0.0 ? dms : 0.9 * ema_disp_ms + 0.1 * dms;

                // Latch the most recent detection result onto this live frame.
                std::vector<ot::Track> tracks;
                ot::TargetState target;
                bool r_locked = false, r_roi = false;
                int64_t r_idx = -1;
                double r_infer_ms = 0.0, r_det_fps = 0.0;
                const bool have_res = result.get(tracks, target, r_locked, r_roi,
                                                 r_idx, r_infer_ms, r_det_fps);
                if (have_res) {
                    ema_ms = ema_ms == 0.0 ? r_infer_ms : 0.9 * ema_ms + 0.1 * r_infer_ms;
                    last_det_fps = r_det_fps;
                }

                // Grab the clean (un-annotated) crop around the locked target for
                // the zoom panel BEFORE overlays are drawn, so the magnified view
                // shows the actual object rather than chunky overlay lines.
                cv::Mat zoom_crop;
                cv::Rect zoom_src;
                if (zoom && have_res && r_locked) {
                    zoom_src = ot::zoom_crop_rect(target.box, W, H);
                    if (zoom_src.area() > 0) zoom_crop = df.img(zoom_src).clone();
                }

                const auto td0 = std::chrono::steady_clock::now();
                if (have_res) {
                    // Dim the background detections only while we actually have a
                    // target to emphasize (Locked/Coasting). On LOST, r_locked stays
                    // true so the red "searching" reticle/HUD/sink keep going, but the
                    // detections must go back to full color — there's nothing to dim
                    // them under, and the user needs them bright to re-acquire.
                    ot::draw_tracks(df.img, tracks, class_map, /*thin=*/target.valid());
                    if (r_locked) reticle.draw(df.img, target, class_map, now_s);
                }

                char hud[256];
                const char* hint = !have_res ? "warming up"
                                 : target.state == ot::LockState::Lost
                                       ? "LOST - searching (click/r)"
                                       : r_locked ? (r_roi ? "LOCK (roi)" : "LOCK (full)")
                                                  : "click a target to lock";
                std::snprintf(hud, sizeof(hud),
                              "frame %lld  trk %zu  infer %.0f ms  det %.0f fps  disp %.1f fps  %s%s",
                              static_cast<long long>(df.index), tracks.size(), ema_ms, last_det_fps,
                              ema_disp_ms > 0 ? 1000.0 / ema_disp_ms : 0.0,
                              hint, paused.load() ? "  [PAUSED]" : "");
                cv::putText(df.img, hud, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

                // Drone telemetry for THIS display frame (not the inference
                // frame): GPS, altitude, per-frame focal, attitude if the telemetry
                // carries it. Telemetry is read-only after load, so reading it
                // here off the inference thread is safe.
                if (telemetry) {
                    if (const ot::TeleFrame* tf = telemetry->at(df.index)) {
                        char tl[192];
                        int n = 0;
                        if (tf->has_gps)
                            n = std::snprintf(tl, sizeof(tl), "GPS %.6f, %.6f  ",
                                              tf->lat, tf->lon);
                        n += std::snprintf(tl + n, sizeof(tl) - n, "alt %.1fm",
                                           tf->rel_alt);
                        if (tf->has_focal)
                            n += std::snprintf(tl + n, sizeof(tl) - n, "  f %.0fmm",
                                               tf->focal_len);
                        if (tf->has_attitude)
                            std::snprintf(tl + n, sizeof(tl) - n, "  gimbal %.1f/%.1f/%.1f",
                                          tf->gb_yaw, tf->gb_pitch, tf->gb_roll);
                        cv::putText(df.img, tl, {12, 54}, cv::FONT_HERSHEY_SIMPLEX,
                                    0.55, {0, 255, 0}, 1, cv::LINE_AA);
                    }
                }
                const double draw_ms = dur_ms(td0, std::chrono::steady_clock::now());
                ema_draw = ema_draw == 0.0 ? draw_ms : 0.9 * ema_draw + 0.1 * draw_ms;

                if (recorder) recorder->write(df.img);   // record the W x H main frame only

                // Compose main frame + zoom panel side by side for display. The
                // panel shows a placeholder when nothing is locked, so the output
                // size stays constant (no window flicker) and screenshots ('s')
                // capture the full view.
                if (zoom) {
                    cv::Mat panel = ot::render_zoom_panel(zoom_crop, zoom_src, target,
                                                          have_res && r_locked, class_map, zoom_w, H);
                    cv::Mat canvas(H, disp_w, df.img.type());
                    df.img.copyTo(canvas(cv::Rect(0, 0, W, H)));
                    panel.copyTo(canvas(cv::Rect(W, 0, zoom_w, H)));
                    df.img = std::move(canvas);
                }
                shown = std::move(df);
                have_shown = true;
            }

            // Re-show the last frame on timeout so the window stays live (e.g.
            // while paused) and keys remain responsive.
            const auto ts0 = std::chrono::steady_clock::now();
            if (have_shown) cv::imshow(kWindow, shown.img);
            cv::Point2f click;
            // Ignore clicks that land in the zoom side panel (x >= W).
            if (selector.poll(click) && click.x < static_cast<float>(W)) cmds.post_click(click);
            const double show_ms = dur_ms(ts0, std::chrono::steady_clock::now());
            ema_show = ema_show == 0.0 ? show_ms : 0.9 * ema_show + 0.1 * show_ms;

            // Pace: hold this frame until its scheduled slot. waitKey is both the
            // wait and the GUI/key pump (returns early on a keypress). The bounded
            // disp_q buffers decode-time variance so the cadence stays steady.
            int pace_ms = 1;
            if (interval_ms > 0.0) {
                const double since = dur_ms(last_present, std::chrono::steady_clock::now());
                if (since < interval_ms) pace_ms = std::max(1, static_cast<int>(std::lround(interval_ms - since)));
            }
            const int key = cv::waitKey(pace_ms) & 0xFF;
            last_present = std::chrono::steady_clock::now();

            if (fresh && profile && ++prof_n % 120 == 0) {
                std::fprintf(stderr,
                    "[profile] display %.1f fps | capture %.1f  infer %.1f (det %.0f fps)  draw %.1f  show %.1f ms\n",
                    ema_disp_ms > 0 ? 1000.0 / ema_disp_ms : 0.0,
                    cap_ms.load(), ema_ms, last_det_fps, ema_draw, ema_show);
            }

            if (key == 'q' || key == 27) { running.store(false); break; }
            if (key == ' ') paused.store(!paused.load());
            if (key == 'r') cmds.post_reset();
            if (key == 's' && have_shown) {
                const std::string file = "shot_" + std::to_string(shown.index) + ".png";
                cv::imwrite(file, shown.img);
                std::printf("[run] saved %s\n", file.c_str());
            }
        }

        // ---- shutdown: wake every blocked stage, then join -----------------
        running.store(false);
        infer_q.close();
        disp_q.close();
        capture.join();
        infer.join();

        if (recorder) recorder->release();
        cv::destroyAllWindows();
        std::printf("[run] stopped\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
