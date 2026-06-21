#include "ot/bounded_queue.hpp"
#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "ot/lock_manager.hpp"
#include "ot/mot_tracker.hpp"
#include "ot/overlay.hpp"
#include "ot/reid.hpp"
#include "ot/selector.hpp"
#include "ot/target_sink.hpp"
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr char kWindow[] = "object_tracking";

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// A decoded working-resolution frame handed from the capture thread to inference.
struct CapturedFrame {
    int64_t  index = 0;
    cv::Mat  img;
};

// Inference output handed to the display thread. Owns its frame outright (the
// frame flows capture -> infer -> display, each stage the sole owner, so there
// is no shared buffer and no clone needed).
struct ShownFrame {
    cv::Mat               img;
    std::vector<ot::Track> tracks;
    ot::TargetState       target;
    bool                  locked = false;
    bool                  roi = false;   // detection ran on the locked ROI fast-path
    int64_t               index = 0;
    double                infer_ms = 0.0;
};

// Predicted search window around the locked target for the ROI fast-path: a
// square centered on the target's next position (current center advanced by one
// frame of its velocity), sized to `scale` x the target's largest dimension and
// clamped to the frame. A tight crop is fine — the detector stretches it up to
// the network input, so a small/distant target actually gains detail.
cv::Rect predict_roi(const ot::TargetState& t, float scale, double fps, int W, int H) {
    cv::Point2f c = t.center_px;
    if (fps > 0) {
        c.x += t.velocity_px_s.x / static_cast<float>(fps);
        c.y += t.velocity_px_s.y / static_cast<float>(fps);
    }
    float side = scale * std::max(t.box.w, t.box.h);
    side = std::max(side, 320.0f);                              // floor: search margin + context
    side = std::min(side, static_cast<float>(std::min(W, H)));  // never exceed the frame
    const int half = static_cast<int>(side * 0.5f);
    const int x0 = std::max(0, static_cast<int>(c.x) - half);
    const int y0 = std::max(0, static_cast<int>(c.y) - half);
    const int x1 = std::min(W, static_cast<int>(c.x) + half);
    const int y1 = std::min(H, static_cast<int>(c.y) + half);
    return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

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
            "  --display-fps N  cap display to N fps for smooth playback (default: 30; 0=uncapped)\n"
            "  --zoom         show a magnified side panel of the locked target\n"
            "  --zoom-width N width of the zoom side panel in px (default: 360)\n"
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
    double display_fps = -1.0;     // -1 = match source fps; 0 = uncapped (benchmark)
    bool zoom = false;             // show a magnified side panel of the locked target
    int  zoom_w = 360;             // zoom side-panel width in px
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--height" && i + 1 < argc)      target_height = std::stoi(argv[++i]);
        else if (a == "--start" && i + 1 < argc)  start_frame = std::stoll(argv[++i]);
        else if (a == "--record" && i + 1 < argc) record_path = argv[++i];
        else if (a == "--loop")                   loop = true;
        else if (a == "--profile")                profile = true;
        else if (a == "--autolock")               autolock = true;
        else if (a == "--display-fps" && i + 1 < argc) display_fps = std::stod(argv[++i]);
        else if (a == "--zoom")                   zoom = true;
        else if (a == "--zoom-width" && i + 1 < argc) zoom_w = std::max(120, std::stoi(argv[++i]));
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

        auto detector = ot::make_detector(cfg.detector, cfg.tiling);
        const ot::ClassMap class_map = ot::ClassMap::preset(cfg.detector.class_map);
        ot::MotTracker tracker(static_cast<int>(fps));

        auto reid = cfg.reid.kind == "onnx"
            ? ot::make_onnx_reid_embedder(cfg.reid.model_path, cfg.reid.input_w, cfg.reid.input_h,
                                          cfg.reid.backend, cfg.reid.device, cfg.reid.precision)
            : ot::make_histogram_embedder();
        std::printf("[run] reid: %s\n", cfg.reid.kind.c_str());
        ot::LockManager lock(fps, cfg.lock.coast_to_lost, cfg.lock.reacquire_thresh, reid,
                             cfg.lock.verify_thresh, cfg.lock.reacquire_max_frac,
                             cfg.lock.reacquire_margin);
        auto sink = ot::make_sink(cfg.output.sink, cfg.output.path);

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
        ot::BoundedQueue<CapturedFrame> cap_q(3);   // capture -> inference
        ot::BoundedQueue<ShownFrame>    disp_q(3);   // inference -> display
        UiCommands         cmds;
        std::atomic<bool>  running{true};
        std::atomic<bool>  paused{false};
        std::atomic<double> cap_ms{0.0};   // capture(decode) stage EMA, for --profile

        // ---- capture thread: decode -> cap_q -------------------------------
        std::thread capture([&] {
            int64_t idx = 0;
            double ema = 0.0;
            cv::Mat frame;
            while (running.load()) {
                if (paused.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = source.read(frame);
                const double rms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                ema = ema == 0.0 ? rms : 0.9 * ema + 0.1 * rms;
                cap_ms.store(ema);
                if (!ok) {                                  // end of stream
                    if (loop && source.seek(0)) { cmds.post_reset(); continue; }
                    break;
                }
                idx = source.frame_index();
                if (!cap_q.push({idx, std::move(frame)})) break;  // closed -> quitting
                frame = cv::Mat();                          // fresh buffer for next read
            }
            cap_q.close();
        });

        // ---- inference thread: cap_q -> detect/track/lock -> disp_q --------
        std::thread infer([&] {
            CapturedFrame in;
            ot::TargetState last_target;     // previous frame's lock state, drives the ROI crop
            int64_t infer_frame = 0;
            const int full_every = std::max(1, cfg.lock.roi_full_interval);
            while (running.load()) {
                if (!cap_q.pop(in)) break;                  // closed + drained

                bool do_reset = false, do_click = false;
                cv::Point2f click;
                cmds.drain(do_reset, do_click, click);
                if (do_reset) { lock.reset(); last_target = ot::TargetState{}; }

                // Locked-ROI fast-path: while solidly locked, detect only a crop
                // around the predicted target (cheap) — except every Nth frame,
                // when full SAHI runs to catch new objects and correct drift. Any
                // non-Locked state (Coasting/Lost/Acquiring) also takes the full
                // path, so a target that slips out of the crop self-recovers.
                const bool use_roi = cfg.lock.roi_fastpath
                                  && last_target.state == ot::LockState::Locked
                                  && (infer_frame % full_every != 0);

                const auto t0 = std::chrono::steady_clock::now();
                std::vector<ot::Detection> dets;
                if (use_roi) {
                    const cv::Rect roi = predict_roi(last_target, cfg.lock.roi_scale, fps, W, H);
                    auto local = detector->detect(in.img(roi));
                    dets.reserve(local.size());
                    for (ot::Detection d : local) {         // shift crop-space boxes to frame space
                        d.box.x += roi.x;
                        d.box.y += roi.y;
                        dets.push_back(d);
                    }
                } else {
                    dets = detector->detect(in.img);        // full-frame SAHI
                }
                auto tracks = tracker.update(in.img, dets);
                const double infer_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

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
                last_target = target;
                ++infer_frame;

                const bool locked = lock.has_target() || lock.state() == ot::LockState::Lost;
                if (locked) {
                    target.frame_index = in.index;
                    target.timestamp_s = fps > 0 ? in.index / fps : 0.0;
                    target.center_norm = {target.center_px.x / W, target.center_px.y / H};
                    sink->write(target);
                }

                ShownFrame out;
                out.img = std::move(in.img);
                out.tracks = std::move(tracks);
                out.target = target;
                out.locked = locked;
                out.roi = use_roi;
                out.index = in.index;
                out.infer_ms = infer_ms;
                if (!disp_q.push(std::move(out))) break;    // closed -> quitting
            }
            disp_q.close();
        });

        // ---- display + UI: main thread (HighGUI must run here) -------------
        auto ms = [](std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        ShownFrame shown;
        bool have_shown = false;
        double ema_ms = 0.0, ema_disp_ms = 0.0, ema_draw = 0.0, ema_show = 0.0;
        auto last_show = std::chrono::steady_clock::now();
        long prof_n = 0;

        // Frame pacing: present at a steady rate so playback is smooth instead of
        // rushing on cheap ROI frames (~90fps) and lurching on full-SAHI frames
        // (~45fps). Default caps to a steady 30 fps — comfortably under the
        // worst-case (unlocked full-SAHI) sustainable rate, so it's smooth in
        // every mode. Raise with --display-fps N; --display-fps 0 uncaps.
        const double eff_fps = display_fps < 0.0 ? std::min(30.0, fps) : display_fps;
        const double interval_ms = eff_fps > 0.0 ? 1000.0 / eff_fps : 0.0;
        auto last_present = std::chrono::steady_clock::now();
        if (interval_ms > 0.0)
            std::printf("[run] display paced to %.1f fps (use --display-fps 0 to uncap)\n", eff_fps);

        while (running.load()) {
            ShownFrame d;
            const int got = disp_q.pop_for(d, std::chrono::milliseconds(30));
            if (got == -1) break;                           // pipeline drained (EOF)

            const bool fresh = (got == 1);
            if (fresh) {
                const auto now = std::chrono::steady_clock::now();
                const double dms = ms(last_show, now);
                last_show = now;
                ema_disp_ms = ema_disp_ms == 0.0 ? dms : 0.9 * ema_disp_ms + 0.1 * dms;
                ema_ms = ema_ms == 0.0 ? d.infer_ms : 0.9 * ema_ms + 0.1 * d.infer_ms;

                // Grab the clean (un-annotated) crop around the locked target for
                // the zoom panel BEFORE overlays are drawn, so the magnified view
                // shows the actual object rather than chunky overlay lines.
                cv::Mat zoom_crop;
                cv::Rect zoom_src;
                if (zoom && d.locked) {
                    zoom_src = ot::zoom_crop_rect(d.target.box, W, H);
                    if (zoom_src.area() > 0) zoom_crop = d.img(zoom_src).clone();
                }

                const auto td0 = std::chrono::steady_clock::now();
                ot::draw_tracks(d.img, d.tracks, class_map, /*thin=*/d.locked);
                if (d.locked) ot::draw_locked_target(d.img, d.target, class_map);

                char hud[224];
                const char* hint = d.target.state == ot::LockState::Lost
                                       ? "LOST - searching (click/r)"
                                       : d.locked ? (d.roi ? "LOCK (roi)" : "LOCK (full)")
                                                  : "click a target to lock";
                std::snprintf(hud, sizeof(hud),
                              "frame %lld  trk %zu  infer %.0f ms  %.1f fps  %s%s",
                              static_cast<long long>(d.index), d.tracks.size(), ema_ms,
                              ema_disp_ms > 0 ? 1000.0 / ema_disp_ms : 0.0,
                              hint, paused.load() ? "  [PAUSED]" : "");
                cv::putText(d.img, hud, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
                const double draw_ms = ms(td0, std::chrono::steady_clock::now());
                ema_draw = ema_draw == 0.0 ? draw_ms : 0.9 * ema_draw + 0.1 * draw_ms;

                if (recorder) recorder->write(d.img);   // record the W x H main frame only
                shown = std::move(d);
                have_shown = true;

                // Compose main frame + zoom panel side by side for display. The
                // panel shows a placeholder when nothing is locked, so the output
                // size stays constant (no window flicker) and screenshots ('s')
                // capture the full view.
                if (zoom) {
                    cv::Mat panel = ot::render_zoom_panel(zoom_crop, zoom_src, shown.target,
                                                          shown.locked, class_map, zoom_w, H);
                    cv::Mat canvas(H, disp_w, shown.img.type());
                    shown.img.copyTo(canvas(cv::Rect(0, 0, W, H)));
                    panel.copyTo(canvas(cv::Rect(W, 0, zoom_w, H)));
                    shown.img = std::move(canvas);
                }
            }

            // Re-show the last frame on timeout so the window stays live (e.g.
            // while paused) and keys remain responsive.
            const auto ts0 = std::chrono::steady_clock::now();
            if (have_shown) cv::imshow(kWindow, shown.img);
            cv::Point2f click;
            // Ignore clicks that land in the zoom side panel (x >= W).
            if (selector.poll(click) && click.x < static_cast<float>(W)) cmds.post_click(click);
            const double show_ms = ms(ts0, std::chrono::steady_clock::now());
            ema_show = ema_show == 0.0 ? show_ms : 0.9 * ema_show + 0.1 * show_ms;

            // Pace: hold this frame until its scheduled slot. waitKey is both the
            // wait and the GUI/key pump (returns early on a keypress). The bounded
            // queues buffer inference-time variance so the cadence stays steady.
            int pace_ms = 1;
            if (interval_ms > 0.0) {
                const double since = ms(last_present, std::chrono::steady_clock::now());
                if (since < interval_ms) pace_ms = std::max(1, static_cast<int>(std::lround(interval_ms - since)));
            }
            const int key = cv::waitKey(pace_ms) & 0xFF;
            last_present = std::chrono::steady_clock::now();

            if (fresh && profile && ++prof_n % 120 == 0) {
                std::fprintf(stderr,
                    "[profile] pipeline %.1f fps | capture %.1f  infer %.1f  draw %.1f  show %.1f ms\n",
                    ema_disp_ms > 0 ? 1000.0 / ema_disp_ms : 0.0,
                    cap_ms.load(), ema_ms, ema_draw, ema_show);
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
        cap_q.close();
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
