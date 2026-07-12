// Logic-only tests for the telemetry + range-estimation stack (no video/GPU):
// SRT parsing (Mavic and enterprise gb_* variants, CRLF, zoom ramp), sidecar
// detection, and per-frame lookup clamping. Optionally parses a real SRT.
//
//   ./build/test_range [path/to/DJI_xxxx.SRT]
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
}  // namespace

int main(int argc, char** argv) {
    std::printf("== SRT parsing: Mavic Fly-app format (no attitude) ==\n");
    {
        // Two blocks in the real Mavic 4 Pro layout, CRLF line endings, with a
        // zoom ramp (70 -> 168 mm) across frames.
        const std::string srt =
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
        const auto frames = ot::parse_srt(write_temp("ot_test_mavic.srt", srt));
        check("two records", frames.size() == 2);
        if (frames.size() == 2) {
            const auto& f0 = frames[0];
            check("FrameCnt", f0.frame == 1 && frames[1].frame == 2);
            check("timestamp", f0.time == "2026-04-25 13:17:52.266", f0.time);
            check("gps", f0.has_gps && near(f0.lat, 52.150959) && near(f0.lon, 21.094026));
            check("rel/abs alt", near(f0.rel_alt, 63.928) && near(f0.abs_alt, 181.211));
            check("zoom ramp per frame", f0.has_focal && near(f0.focal_len, 70.0) &&
                                          near(frames[1].focal_len, 168.0));
            check("no attitude in Fly-app SRT", !f0.has_attitude && !frames[1].has_attitude);
        }
    }

    std::printf("== SRT parsing: enterprise format (gb_* attitude) ==\n");
    {
        const std::string srt =
            "1\n"
            "00:00:00,000 --> 00:00:00,033\n"
            "<font size=\"28\">FrameCnt: 1, DiffTime: 33ms\n"
            "2026-04-25 10:00:00.000\n"
            "[focal_len: 24.00] [latitude: 52.1] [longitude: 21.0] "
            "[rel_alt: 50.000 abs_alt: 150.000] "
            "[gb_yaw: -12.3 gb_pitch: -45.6 gb_roll: 0.4] </font>\n"
            "\n";
        const auto frames = ot::parse_srt(write_temp("ot_test_ent.srt", srt));
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
              ot::find_srt_sidecar("/nonexistent/video.mp4").empty());
        // A real video file with an .SRT next to it must resolve; simulate one.
        const auto dir = std::filesystem::temp_directory_path();
        const std::string vid = (dir / "ot_test_clip.MP4").string();
        const std::string srt = (dir / "ot_test_clip.SRT").string();
        std::ofstream(vid) << "x";
        std::ofstream(srt) << "x";
        check("finds .SRT next to video", ot::find_srt_sidecar(vid) == srt,
              ot::find_srt_sidecar(vid));
        check("rtsp url -> no sidecar",
              ot::find_srt_sidecar("rtsp://cam/stream").empty());
    }

    if (argc > 1) {
        std::printf("== real SRT: %s ==\n", argv[1]);
        try {
            const auto frames = ot::parse_srt(argv[1]);
            const auto& f0 = frames.front();
            const auto& fN = frames.back();
            char d[256];
            std::snprintf(d, sizeof(d),
                          "%zu records; first: frame=%d t=%s gps=%.6f,%.6f alt=%.1f "
                          "f=%.1fmm att=%d; last: frame=%d f=%.1fmm",
                          frames.size(), f0.frame, f0.time.c_str(), f0.lat, f0.lon,
                          f0.rel_alt, f0.focal_len, f0.has_attitude ? 1 : 0,
                          fN.frame, fN.focal_len);
            check("SRT parses", !frames.empty(), d);
            check("first record has GPS + focal", f0.has_gps && f0.has_focal);
            check("FrameCnt is 1-based and contiguous",
                  f0.frame == 1 && fN.frame == static_cast<int>(frames.size()));
        } catch (const std::exception& e) {
            check("SRT parses", false, e.what());
        }
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
