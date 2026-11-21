// Logic-only tests for the telemetry + range-estimation stack (no video/GPU):
// telemetry parsing (basic and gb_* attitude variants, CRLF, zoom ramp), sidecar
// detection, per-frame lookup clamping, and the RangeEstimator method chain
// (size-prior math, smoothing, resets, guards). Optionally parses a real telemetry file.
//
//   ./build/test_range [path/to/telemetry_file]
#include "ot/range_estimator.hpp"
#include "ot/telemetry.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(const char* name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++failures;
}

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) < tol; }

// Writes content to a temp file and returns its path.
std::string write_temp(const char* name, const std::string& content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::trunc | std::ios::binary);
    out << content;
    return path.string();
}

// A locked target of the given class whose box's longest dimension is dim_px.
ot::TargetState locked_target(int class_id, float dim_px, int lock_id = 1) {
    ot::TargetState t;
    t.state = ot::LockState::Locked;
    t.class_id = class_id;
    t.lock_id = lock_id;
    t.box = {100.0f, 100.0f, dim_px, dim_px * 0.4f};
    return t;
}

// A locked target whose box BOTTOM-CENTER (the ground-contact pixel the
// ground-plane method casts through) sits exactly at (u, v).
ot::TargetState target_at(float u, float v, int class_id = 2, int lock_id = 1) {
    ot::TargetState t;
    t.state = ot::LockState::Locked;
    t.class_id = class_id;
    t.lock_id = lock_id;
    t.box = {u - 20.0f, v - 16.0f, 40.0f, 16.0f};
    return t;
}

ot::TeleFrame tele_frame(double focal_mm, double rel_alt) {
    ot::TeleFrame f;
    f.focal_len = focal_mm;
    f.has_focal = true;
    f.rel_alt = rel_alt;
    return f;
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("== telemetry parsing: basic format (no attitude) ==\n");
    {
        // Two blocks in a real consumer-drone layout, CRLF line endings, with a
        // zoom ramp (70 -> 168 mm) across frames.
        const std::string tele =
            "1\r\n"
            "00:00:00,000 --> 00:00:00,016\r\n"
            "<font size=\"28\">FrameCnt: 1, DiffTime: 16ms\r\n"
            "2026-04-25 13:17:52.266\r\n"
            "[iso: 100] [shutter: 1/240.0] [fnum: 2.8] [ev: 0] [color_md: default] "
            "[focal_len: 70.00] [latitude: 52.150959] [longitude: 21.094026] "
            "[rel_alt: 63.928 abs_alt: 181.211] [ct: 5438, tint: 10] </font>\r\n"
            "\r\n"
            "2\r\n"
            "00:00:00,016 --> 00:00:00,032\r\n"
            "<font size=\"28\">FrameCnt: 2, DiffTime: 16ms\r\n"
            "2026-04-25 13:17:52.283\r\n"
            "[iso: 100] [focal_len: 168.00] [latitude: 52.150960] [longitude: 21.094027] "
            "[rel_alt: 64.100 abs_alt: 181.400] </font>\r\n"
            "\r\n";
        const auto frames = ot::parse_telemetry(write_temp("ot_test_tele_basic.txt", tele));
        check("two records", frames.size() == 2);
        if (frames.size() == 2) {
            const auto& f0 = frames[0];
            check("FrameCnt", f0.frame == 1 && frames[1].frame == 2);
            check("timestamp", f0.time == "2026-04-25 13:17:52.266", f0.time);
            check("gps", f0.has_gps && near(f0.lat, 52.150959) && near(f0.lon, 21.094026));
            check("rel/abs alt", near(f0.rel_alt, 63.928) && near(f0.abs_alt, 181.211));
            check("zoom ramp per frame", f0.has_focal && near(f0.focal_len, 70.0) &&
                                          near(frames[1].focal_len, 168.0));
            check("no attitude in basic telemetry", !f0.has_attitude && !frames[1].has_attitude);
        }
    }

    std::printf("== telemetry parsing: with gimbal attitude (gb_*) ==\n");
    {
        const std::string tele =
            "1\n"
            "00:00:00,000 --> 00:00:00,033\n"
            "<font size=\"28\">FrameCnt: 1, DiffTime: 33ms\n"
            "2026-04-25 10:00:00.000\n"
            "[focal_len: 24.00] [latitude: 52.1] [longitude: 21.0] "
            "[rel_alt: 50.000 abs_alt: 150.000] "
            "[gb_yaw: -12.3 gb_pitch: -45.6 gb_roll: 0.4] </font>\n"
            "\n";
        const auto frames = ot::parse_telemetry(write_temp("ot_test_tele_gimbal.txt", tele));
        check("record parses", frames.size() == 1);
        if (frames.size() == 1) {
            const auto& f = frames[0];
            check("has attitude", f.has_attitude);
            check("gimbal values", near(f.gb_yaw, -12.3) && near(f.gb_pitch, -45.6) &&
                                    near(f.gb_roll, 0.4));
        }
    }

    std::printf("== Telemetry lookup ==\n");
    {
        std::vector<ot::TeleFrame> recs(3);
        for (int i = 0; i < 3; ++i) recs[i].frame = i + 1;
        const ot::Telemetry tele(std::move(recs));
        check("index 0 -> FrameCnt 1", tele.at(0) && tele.at(0)->frame == 1);
        check("clamps past the end", tele.at(999) && tele.at(999)->frame == 3);
        check("negative index -> null", tele.at(-1) == nullptr);
        check("no attitude anywhere", !tele.has_any_attitude());
        check("empty telemetry -> null", ot::Telemetry({}).at(0) == nullptr);
    }

    std::printf("== Sidecar detection ==\n");
    {
        check("missing video -> no sidecar",
              ot::find_telemetry_sidecar("/nonexistent/video.mp4").empty());
        // A real video file with a telemetry file next to it must resolve; simulate one.
        const auto dir = std::filesystem::temp_directory_path();
        const std::string vid = (dir / "ot_test_clip.MP4").string();
        const std::string tele = (dir / "ot_test_clip.SRT").string();
        std::ofstream(vid) << "x";
        std::ofstream(tele) << "x";
        check("finds telemetry file next to video", ot::find_telemetry_sidecar(vid) == tele,
              ot::find_telemetry_sidecar(vid));
        check("rtsp url -> no sidecar",
              ot::find_telemetry_sidecar("rtsp://cam/stream").empty());
    }

    std::printf("== RangeEstimator: size-prior method ==\n");
    {
        const ot::ClassMap cmap = ot::ClassMap::preset("coco");  // car = 2, person = 0
        ot::RangeCfg cfg;                                        // defaults: car 4.6m
        ot::RangeEstimator est(cfg, 1920, 1080, cmap);

        // f_px = 168/36*1920 = 8960; a 4.6m car spanning 137.3867px sits at 300m.
        const ot::TeleFrame tf = tele_frame(168.0, 71.0);
        const auto r = est.update(locked_target(2, 137.3867f), &tf);
        check("method is size-prior", r.method == ot::RangeMethod::SizePrior);
        check("slant ~300m", near(r.slant_m, 300.0, 0.01),
              std::to_string(r.slant_m));
        // Flat ground at 71m AGL: ground = sqrt(300^2 - 71^2) = 291.48m.
        check("ground ~291.5m", near(r.ground_m, 291.477, 0.01),
              std::to_string(r.ground_m));

        // Class without a prior (airplane isn't a kept class) -> no estimate.
        ot::RangeEstimator est2(cfg, 1920, 1080, cmap);
        check("no prior class -> none",
              est2.update(locked_target(4, 100.0f), &tf).method == ot::RangeMethod::None);

        // No focal from telemetry and no override -> no estimate.
        ot::TeleFrame no_focal;
        no_focal.rel_alt = 71.0;
        ot::RangeEstimator est3(cfg, 1920, 1080, cmap);
        check("no focal -> none",
              est3.update(locked_target(2, 100.0f), &no_focal).method == ot::RangeMethod::None);
        check("no telemetry -> none",
              est3.update(locked_target(2, 100.0f), nullptr).method == ot::RangeMethod::None);

        // equiv_focal_mm override enables the size branch without any telemetry:
        // f_px = 70/36*1920 = 3733.33; a 1.7m person at 21.156px sits at 300m.
        ot::RangeCfg cfg_f = cfg;
        cfg_f.equiv_focal_mm = 70.0f;
        ot::RangeEstimator est4(cfg_f, 1920, 1080, cmap);
        const auto rf = est4.update(locked_target(0, 21.1556f), nullptr);
        check("focal override works without telemetry",
              rf.method == ot::RangeMethod::SizePrior && near(rf.slant_m, 300.0, 0.05),
              std::to_string(rf.slant_m));
        check("ground unknown without altitude", rf.ground_m < 0.0f);

        // Implausibly far (2px box -> 20.6km) exceeds max_range_m -> rejected.
        ot::RangeEstimator est5(cfg, 1920, 1080, cmap);
        check("max range guard",
              est5.update(locked_target(2, 2.0f), &tf).method == ot::RangeMethod::None);

        // Lost target: the box is a stale prediction -> no estimate.
        ot::TargetState lost = locked_target(2, 137.3867f);
        lost.state = ot::LockState::Lost;
        ot::RangeEstimator est6(cfg, 1920, 1080, cmap);
        check("lost -> none", est6.update(lost, &tf).method == ot::RangeMethod::None);
        ot::TargetState coasting = locked_target(2, 137.3867f);
        coasting.state = ot::LockState::Coasting;
        check("coasting still estimates",
              est6.update(coasting, &tf).method == ot::RangeMethod::SizePrior);

        // range.enabled: false -> never estimates.
        ot::RangeCfg off = cfg;
        off.enabled = false;
        ot::RangeEstimator est7(off, 1920, 1080, cmap);
        check("disabled -> none",
              est7.update(locked_target(2, 137.3867f), &tf).method == ot::RangeMethod::None);
    }

    std::printf("== RangeEstimator: smoothing & resets ==\n");
    {
        const ot::ClassMap cmap = ot::ClassMap::preset("coco");
        ot::RangeCfg cfg;
        ot::RangeEstimator est(cfg, 1920, 1080, cmap);
        const ot::TeleFrame tf = tele_frame(168.0, 71.0);

        // Slants 100 / 500 / 100 (dims 412.16 / 82.432 / 412.16 px): the rolling
        // median must reject the one-frame 500m spike.
        est.update(locked_target(2, 412.16f), &tf);
        est.update(locked_target(2, 82.432f), &tf);
        const auto r = est.update(locked_target(2, 412.16f), &tf);
        check("median rejects the spike", near(r.slant_m, 100.0, 0.01),
              std::to_string(r.slant_m));

        // A new lock id starts a fresh window: the first estimate is its own.
        const auto r2 = est.update(locked_target(2, 82.432f, /*lock_id=*/2), &tf);
        check("new lock resets the window", near(r2.slant_m, 500.0, 0.05),
              std::to_string(r2.slant_m));

        // Explicit reset (r key / loop restart) also clears the window.
        est.update(locked_target(2, 412.16f, 2), &tf);
        est.reset();
        const auto r3 = est.update(locked_target(2, 82.432f, 2), &tf);
        check("reset clears the window", near(r3.slant_m, 500.0, 0.05),
              std::to_string(r3.slant_m));
    }

    std::printf("== RangeEstimator: ground-plane method ==\n");
    {
        const ot::ClassMap cmap = ot::ClassMap::preset("coco");
        ot::RangeCfg cfg;
        const ot::TeleFrame tf = tele_frame(168.0, 100.0);   // 100m AGL

        // Nadir: pitch -90, ray through the image center -> slant = altitude,
        // ground = 0 (the target is directly below the drone).
        ot::RangeEstimator nadir(cfg, 1920, 1080, cmap);
        nadir.set_static_pitch(-90.0f);
        const auto rn = nadir.update(target_at(960.0f, 540.0f), &tf);
        check("nadir method is gp", rn.method == ot::RangeMethod::GroundPlane);
        check("nadir slant = altitude", near(rn.slant_m, 100.0, 0.01),
              std::to_string(rn.slant_m));
        check("nadir ground ~0", rn.ground_m >= 0.0f && rn.ground_m < 0.1f,
              std::to_string(rn.ground_m));

        // 45 deg depression, center ray: slant = h/sin45 = 141.42, ground = 100.
        ot::RangeEstimator o45(cfg, 1920, 1080, cmap);
        o45.set_static_pitch(-45.0f);
        const auto r45 = o45.update(target_at(960.0f, 540.0f), &tf);
        check("45deg slant = h*sqrt(2)", near(r45.slant_m, 141.4214, 0.01),
              std::to_string(r45.slant_m));
        check("45deg ground = h", near(r45.ground_m, 100.0, 0.01),
              std::to_string(r45.ground_m));

        // Pixel offset: a point 15deg below the optical axis at pitch -45 sits
        // at 60deg depression -> slant = h/sin60, ground = h/tan60. The pixel
        // is v = cy + f_px*tan(15deg) with f_px = 168/36*1920 = 8960.
        const float v_off = 540.0f + 8960.0f * std::tan(15.0 * M_PI / 180.0);
        ot::RangeEstimator o60(cfg, 1920, 1080, cmap);
        o60.set_static_pitch(-45.0f);
        const auto r60 = o60.update(target_at(960.0f, v_off), &tf);
        check("pixel offset adds to depression",
              near(r60.slant_m, 100.0 / std::sin(60.0 * M_PI / 180.0), 0.05) &&
              near(r60.ground_m, 100.0 / std::tan(60.0 * M_PI / 180.0), 0.05),
              std::to_string(r60.slant_m) + " / " + std::to_string(r60.ground_m));

        // telemetry gb_* attitude drives gp without any static pitch...
        ot::TeleFrame tf_att = tf;
        tf_att.has_attitude = true;
        tf_att.gb_pitch = -45.0;
        ot::RangeEstimator tele_att(cfg, 1920, 1080, cmap);
        const auto ra = tele_att.update(target_at(960.0f, 540.0f), &tf_att);
        check("telemetry attitude -> gp", ra.method == ot::RangeMethod::GroundPlane &&
                                     near(ra.slant_m, 141.4214, 0.01));
        // ...and roll about the optical axis leaves the center ray unchanged.
        ot::TeleFrame tf_roll = tf_att;
        tf_roll.gb_roll = 2.0;
        ot::RangeEstimator rolled(cfg, 1920, 1080, cmap);
        const auto rr = rolled.update(target_at(960.0f, 540.0f), &tf_roll);
        check("roll leaves the center ray unchanged", near(rr.slant_m, 141.4214, 0.01));

        // --gimbal-pitch overrides the telemetry attitude.
        ot::TeleFrame tf_bad = tf;
        tf_bad.has_attitude = true;
        tf_bad.gb_pitch = -10.0;
        ot::RangeEstimator prefer_static(cfg, 1920, 1080, cmap);
        prefer_static.set_static_pitch(-45.0f);
        const auto rp = prefer_static.update(target_at(960.0f, 540.0f), &tf_bad);
        check("static pitch beats telemetry attitude", near(rp.slant_m, 141.4214, 0.01),
              std::to_string(rp.slant_m));

        // Guards fall through to the size prior (car box -> est, not gp):
        // near-horizon ray (pitch -0.5deg < min_depression 1.0)...
        ot::RangeEstimator shallow(cfg, 1920, 1080, cmap);
        shallow.set_static_pitch(-0.5f);
        const auto rs = shallow.update(locked_target(2, 137.3867f), &tf);
        check("near-horizon falls back to est", rs.method == ot::RangeMethod::SizePrior);
        // ...ray above the horizon...
        ot::RangeEstimator up(cfg, 1920, 1080, cmap);
        up.set_static_pitch(10.0f);
        check("above horizon falls back to est",
              up.update(locked_target(2, 137.3867f), &tf).method ==
                  ot::RangeMethod::SizePrior);
        // ...and no usable camera height (rel_alt 0).
        const ot::TeleFrame tf_low = tele_frame(168.0, 0.0);
        ot::RangeEstimator noalt(cfg, 1920, 1080, cmap);
        noalt.set_static_pitch(-45.0f);
        check("no altitude falls back to est",
              noalt.update(locked_target(2, 137.3867f), &tf_low).method ==
                  ot::RangeMethod::SizePrior);

        // gp beats the size prior when both are available.
        ot::RangeEstimator both(cfg, 1920, 1080, cmap);
        both.set_static_pitch(-45.0f);
        check("gp wins over est",
              both.update(target_at(960.0f, 540.0f, /*car*/ 2), &tf).method ==
                  ot::RangeMethod::GroundPlane);

        // Method change (est -> gp via appearing telemetry attitude) resets the
        // median window, so the first gp value stands alone.
        ot::RangeEstimator sw(cfg, 1920, 1080, cmap);
        sw.update(locked_target(2, 137.3867f), &tf);       // est ~300m
        sw.update(locked_target(2, 137.3867f), &tf);       // est ~300m
        const auto rsw = sw.update(target_at(960.0f, 540.0f), &tf_att);  // gp 141.4m
        check("method switch resets the window",
              rsw.method == ot::RangeMethod::GroundPlane && near(rsw.slant_m, 141.4214, 0.01),
              std::to_string(rsw.slant_m));
    }

    if (argc > 1) {
        std::printf("== real telemetry file: %s ==\n", argv[1]);
        try {
            const auto frames = ot::parse_telemetry(argv[1]);
            const auto& f0 = frames.front();
            const auto& fN = frames.back();
            char d[256];
            std::snprintf(d, sizeof(d),
                          "%zu records; first: frame=%d t=%s gps=%.6f,%.6f alt=%.1f "
                          "f=%.1fmm att=%d; last: frame=%d f=%.1fmm",
                          frames.size(), f0.frame, f0.time.c_str(), f0.lat, f0.lon,
                          f0.rel_alt, f0.focal_len, f0.has_attitude ? 1 : 0,
                          fN.frame, fN.focal_len);
            check("telemetry file parses", !frames.empty(), d);
            check("first record has GPS + focal", f0.has_gps && f0.has_focal);
            check("FrameCnt is 1-based and contiguous",
                  f0.frame == 1 && fN.frame == static_cast<int>(frames.size()));
        } catch (const std::exception& e) {
            check("telemetry file parses", false, e.what());
        }
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
