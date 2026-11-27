// Logic-only tests for the map feature (no GPU, no network by default):
// embedded metadata-track decoding, camera-heading resolution, telemetry merge,
// target position from range + bearing, map/tile math, the tile cache's disk
// path, and the map panel's trail/zoom bookkeeping rendered offline.
//
//   ./build/test_geo_map [video_file]
//
// With a video file it also reads that file's embedded metadata track and
// prints what it found. Environment switches for eyeballing the panel:
//   OT_TEST_MAP_PNG=out.png     save the offline panel render
//   OT_TEST_MAP_ONLINE=out.png  render a fixed public landmark WITH tile
//                               downloads (uses the network) and save it
#include "ot/geodesy.hpp"
#include "ot/map_panel.hpp"
#include "ot/range_estimator.hpp"
#include "ot/telemetry.hpp"
#include "ot/tile_cache.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;

void check(const char* name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++failures;
}

bool near(double a, double b, double tol) { return std::fabs(a - b) < tol; }

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---- tiny protobuf writer, to build metadata packets for the decoder -------
using Bytes = std::vector<uint8_t>;

void put_varint(Bytes& b, uint64_t v) {
    while (v >= 0x80) {
        b.push_back(static_cast<uint8_t>(v) | 0x80);
        v >>= 7;
    }
    b.push_back(static_cast<uint8_t>(v));
}
void put_int(Bytes& b, uint32_t field, int64_t v) {
    put_varint(b, (field << 3) | 0);
    put_varint(b, static_cast<uint64_t>(v));   // negatives: 10-byte two's complement
}
void put_msg(Bytes& b, uint32_t field, const Bytes& m) {
    put_varint(b, (field << 3) | 2);
    put_varint(b, m.size());
    b.insert(b.end(), m.begin(), m.end());
}
void put_f64(Bytes& b, uint32_t field, double v) {
    put_varint(b, (field << 3) | 1);
    uint8_t raw[8];
    std::memcpy(raw, &v, 8);
    b.insert(b.end(), raw, raw + 8);
}
void put_f32(Bytes& b, uint32_t field, float v) {
    put_varint(b, (field << 3) | 5);
    uint8_t raw[4];
    std::memcpy(raw, &v, 4);
    b.insert(b.end(), raw, raw + 4);
}
// Three angles in tenths of a degree; zeros are omitted, as proto3 does.
Bytes angles(int a, int b, int c) {
    Bytes m;
    if (a) put_int(m, 1, a);
    if (b) put_int(m, 2, b);
    if (c) put_int(m, 3, c);
    return m;
}

// One metadata packet in the layout the decoder expects.
Bytes make_packet(int body_yaw_tenths, int gb_pitch_tenths, int gb_roll_tenths,
                  int gb_yaw_tenths, double lat, double lon, double rel_alt_m,
                  double abs_alt_m) {
    Bytes pos;
    put_f64(pos, 2, lat);
    put_f64(pos, 3, lon);
    Bytes gps;
    put_msg(gps, 1, pos);
    put_int(gps, 2, static_cast<int64_t>(std::llround(abs_alt_m * 1000.0)));
    Bytes alt;
    put_f32(alt, 1, static_cast<float>(rel_alt_m * 1000.0));

    Bytes air;
    put_msg(air, 3, angles(-86, -62, body_yaw_tenths));
    put_msg(air, 4, gps);
    put_msg(air, 5, alt);
    Bytes gimbal;
    put_msg(gimbal, 3, angles(gb_pitch_tenths, gb_roll_tenths, gb_yaw_tenths));

    Bytes root;
    put_int(root, 1, 7);                 // unrelated fields are skipped
    put_msg(root, 3, air);
    put_msg(root, 4, gimbal);
    Bytes pkt;
    put_msg(pkt, 1, Bytes{'h', 'd', 'r'});
    put_msg(pkt, 3, root);
    return pkt;
}

ot::TargetState target_at(float u, float v, int class_id = 2, int lock_id = 1) {
    ot::TargetState t;
    t.state = ot::LockState::Locked;
    t.class_id = class_id;
    t.lock_id = lock_id;
    t.box = {u - 20.0f, v - 16.0f, 40.0f, 16.0f};
    return t;
}

ot::TeleFrame tele(double lat, double lon, double rel_alt, double focal_mm) {
    ot::TeleFrame f;
    f.lat = lat;
    f.lon = lon;
    f.has_gps = true;
    f.rel_alt = rel_alt;
    f.focal_len = focal_mm;
    f.has_focal = true;
    return f;
}

std::string temp_dir(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.string();
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("== embedded metadata packet decoding ==\n");
    {
        const Bytes pkt = make_packet(-710, -181, 0, -1238, 52.150959, 21.094030, 63.976, 181.258);
        ot::TeleFrame f;
        check("packet decodes", ot::decode_embedded_packet(pkt.data(), pkt.size(), f));
        check("airframe yaw", f.has_body_yaw && near(f.body_yaw, -71.0, 1e-9), fmt("%.2f", f.body_yaw));
        check("gimbal pitch/roll/yaw",
              f.has_attitude && f.has_gb_yaw && near(f.gb_pitch, -18.1, 1e-9) &&
                  near(f.gb_roll, 0.0, 1e-9) && near(f.gb_yaw, -123.8, 1e-9),
              fmt("%.1f / %.1f / %.1f", f.gb_pitch, f.gb_roll, f.gb_yaw));
        check("gps", f.has_gps && near(f.lat, 52.150959, 1e-9) && near(f.lon, 21.094030, 1e-9));
        check("altitudes", near(f.rel_alt, 63.976, 1e-3) && near(f.abs_alt, 181.258, 1e-6),
              fmt("rel %.3f abs %.3f", f.rel_alt, f.abs_alt));
        check("focal / heading untouched by the decoder", !f.has_focal && !f.has_heading);

        ot::TeleFrame g;
        check("empty packet rejected", !ot::decode_embedded_packet(nullptr, 0, g));
        const uint8_t junk[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01};
        check("garbage rejected", !ot::decode_embedded_packet(junk, sizeof(junk), g));
        bool survived = true;
        for (size_t n = 0; n < pkt.size(); ++n) {   // every truncation must be safe
            ot::TeleFrame t;
            ot::decode_embedded_packet(pkt.data(), n, t);
            if (t.has_gps && !(std::fabs(t.lat) <= 90.0)) survived = false;
        }
        check("truncated packets are safe", survived);

        const Bytes wild = make_packet(99999, 99999, 0, 99999, 52.0, 21.0, 10, 100);
        ot::TeleFrame w;
        ot::decode_embedded_packet(wild.data(), wild.size(), w);
        check("implausible angles are dropped", !w.has_body_yaw && !w.has_attitude && w.has_gps);

        const Bytes nogps = make_packet(-710, -181, 0, -1238, 0.0, 0.0, 10, 100);
        ot::TeleFrame z;
        ot::decode_embedded_packet(nogps.data(), nogps.size(), z);
        check("0,0 position is treated as no fix", !z.has_gps && z.has_attitude);
    }

    std::printf("== camera heading resolution ==\n");
    {
        // Gimbal yaw sits 52 deg behind the airframe yaw; a few frames lag
        // during a turn. The sweep crosses the +-180 wrap.
        std::vector<ot::TeleFrame> recs;
        for (int i = 0; i < 200; ++i) {
            ot::TeleFrame f;
            f.body_yaw = ot::wrap180(150.0 + i * 0.4);          // 150 -> 229.6 (wraps)
            f.has_body_yaw = true;
            const double lag = (i >= 90 && i < 100) ? 9.0 : 0.0;   // transient outliers
            f.gb_yaw = ot::wrap180(f.body_yaw - 52.0 - lag);
            f.has_gb_yaw = true;
            recs.push_back(f);
        }
        const ot::HeadingInfo hi = ot::resolve_heading(recs, kNaN);
        check("offset measured from the clip",
              hi.source == ot::HeadingInfo::Source::GimbalPlusOffset && hi.offset_auto &&
                  near(hi.offset_deg, 52.0, 0.05),
              fmt("offset %.2f", hi.offset_deg));
        check("every record gets a heading", hi.frames == recs.size());
        check("heading follows the airframe yaw",
              recs[10].has_heading && near(ot::wrap180(recs[10].heading - recs[10].body_yaw), 0.0, 0.05) &&
                  near(ot::wrap180(recs[190].heading - recs[190].body_yaw), 0.0, 0.05));
        check("heading follows the GIMBAL while it lags",
              near(ot::wrap180(recs[95].heading - recs[95].body_yaw), -9.0, 0.05));

        const ot::HeadingInfo ov = ot::resolve_heading(recs, 10.0);
        check("explicit offset overrides the measurement",
              !ov.offset_auto && near(ov.offset_deg, 10.0, 1e-9) &&
                  near(ot::wrap180(recs[0].heading - recs[0].gb_yaw), 10.0, 1e-9));

        std::vector<ot::TeleFrame> body(3);
        for (auto& f : body) { f.body_yaw = -71.0; f.has_body_yaw = true; }
        check("airframe yaw alone is used as heading",
              ot::resolve_heading(body, kNaN).source == ot::HeadingInfo::Source::BodyYaw &&
                  near(body[1].heading, -71.0, 1e-9));

        std::vector<ot::TeleFrame> gim(3);
        for (auto& f : gim) { f.gb_yaw = 33.0; f.has_gb_yaw = true; }
        check("gimbal yaw alone is taken as recorded",
              ot::resolve_heading(gim, kNaN).source == ot::HeadingInfo::Source::GimbalYaw &&
                  near(gim[2].heading, 33.0, 1e-9));

        std::vector<ot::TeleFrame> none(3);
        const ot::HeadingInfo no = ot::resolve_heading(none, kNaN);
        check("no yaw anywhere -> no heading",
              no.source == ot::HeadingInfo::Source::None && no.frames == 0 && !none[0].has_heading);
    }

    std::printf("== telemetry merge ==\n");
    {
        std::vector<ot::TeleFrame> base(3), extra(2);
        for (auto& f : base) { f.lat = 52.0; f.lon = 21.0; f.has_gps = true; f.focal_len = 70; f.has_focal = true; }
        base[1].gb_pitch = -30.0;
        base[1].has_attitude = true;
        for (auto& f : extra) {
            f.lat = 1.0; f.lon = 2.0; f.has_gps = true;
            f.gb_pitch = -18.0; f.has_attitude = true;
            f.gb_yaw = -120.0; f.has_gb_yaw = true;
            f.body_yaw = -70.0; f.has_body_yaw = true;
        }
        ot::merge_telemetry(base, extra);
        check("missing attitude is filled in",
              base[0].has_attitude && near(base[0].gb_pitch, -18.0, 1e-9) && base[0].has_body_yaw);
        check("existing values win", near(base[0].lat, 52.0, 1e-9) && near(base[1].gb_pitch, -30.0, 1e-9));
        check("yaw is filled independently of pitch", base[1].has_gb_yaw && near(base[1].gb_yaw, -120.0, 1e-9));
        check("records past the shorter list are untouched", !base[2].has_attitude && base[2].has_focal);
        std::vector<ot::TeleFrame> empty;
        ot::merge_telemetry(empty, extra);
        check("empty base becomes the embedded records", empty.size() == 2 && empty[0].has_body_yaw);
    }

    std::printf("== geodesy ==\n");
    {
        check("wrap180", near(ot::wrap180(190.0), -170.0, 1e-9) && near(ot::wrap180(-190.0), 170.0, 1e-9) &&
                             near(ot::wrap180(180.0), 180.0, 1e-9) && near(ot::wrap180(-180.0), 180.0, 1e-9));
        double lat, lon, e, n;
        ot::destination(52.0, 21.0, 90.0, 100.0, lat, lon);
        ot::enu_offset(52.0, 21.0, lat, lon, e, n);
        check("100 m due east", near(e, 100.0, 1e-6) && near(n, 0.0, 1e-6), fmt("e %.3f n %.3f", e, n));
        ot::destination(52.0, 21.0, 225.0, 200.0, lat, lon);
        ot::enu_offset(52.0, 21.0, lat, lon, e, n);
        check("200 m south-west", near(e, -141.421, 1e-2) && near(n, -141.421, 1e-2));
    }

    std::printf("== target position from range + bearing ==\n");
    {
        const ot::ClassMap cmap = ot::ClassMap::preset("coco");   // car = 2
        ot::RangeCfg cfg;
        cfg.smooth_window = 1;
        const int W = 1920, H = 1080;   // focal 36 mm equiv -> f_px = 1920

        // Camera 100 m up, pitched 45 deg down, looking due east: the centre
        // pixel hits the ground 100 m east of the drone.
        ot::TeleFrame f = tele(52.0, 21.0, 100.0, 36.0);
        f.gb_pitch = -45.0;
        f.has_attitude = true;
        f.heading = 90.0;
        f.has_heading = true;
        ot::RangeEstimator est(cfg, W, H, cmap);
        ot::RangeEstimate r = est.update(target_at(960, 540), &f);
        double e, n;
        ot::enu_offset(f.lat, f.lon, r.lat, r.lon, e, n);
        check("ground-plane: centre pixel, heading east",
              r.has_geo && r.method == ot::RangeMethod::GroundPlane && near(e, 100.0, 0.05) &&
                  near(n, 0.0, 0.05) && near(r.bearing_deg, 90.0, 0.01),
              fmt("e %.2f n %.2f brg %.2f", e, n, r.bearing_deg));

        // Same camera looking north, target 10 deg right of centre: on the
        // ground that is 100 m north and 100*tan(10)/cos(45) = 24.94 m east.
        f.heading = 0.0;
        const float u = 960.0f + 1920.0f * static_cast<float>(std::tan(10.0 * ot::kDegToRad));
        ot::RangeEstimator est2(cfg, W, H, cmap);
        r = est2.update(target_at(u, 540), &f);
        ot::enu_offset(f.lat, f.lon, r.lat, r.lon, e, n);
        check("ground-plane: off-centre pixel", r.has_geo && near(e, 24.94, 0.1) && near(n, 100.0, 0.1) &&
                                                     near(r.bearing_deg, 14.0, 0.1),
              fmt("e %.2f n %.2f brg %.2f", e, n, r.bearing_deg));

        f.heading = -90.0;
        ot::RangeEstimator est3(cfg, W, H, cmap);
        r = est3.update(target_at(960, 540), &f);
        check("bearing is reported in [0,360)", r.has_geo && near(r.bearing_deg, 270.0, 0.01),
              fmt("%.2f", r.bearing_deg));

        f.has_heading = false;
        ot::RangeEstimator est4(cfg, W, H, cmap);
        r = est4.update(target_at(960, 540), &f);
        check("no heading -> a range but no position",
              !r.has_geo && r.method == ot::RangeMethod::GroundPlane && near(r.ground_m, 100.0, 0.05));

        // No attitude at all: size prior gives the slant (car 4.6 m seen
        // 44.16 px wide -> 200 m), altitude turns it into 173.2 m of ground,
        // and the bearing is the heading plus the pixel's horizontal angle.
        ot::TeleFrame s = tele(52.0, 21.0, 100.0, 36.0);
        s.heading = 30.0;
        s.has_heading = true;
        ot::TargetState car;
        car.state = ot::LockState::Locked;
        car.class_id = 2;
        car.lock_id = 1;
        car.box = {960.0f - 22.08f, 500.0f, 44.16f, 20.0f};
        ot::RangeEstimator est5(cfg, W, H, cmap);
        r = est5.update(car, &s);
        ot::enu_offset(s.lat, s.lon, r.lat, r.lon, e, n);
        check("size prior: position along the heading",
              r.has_geo && r.method == ot::RangeMethod::SizePrior && near(std::hypot(e, n), 173.2, 0.2) &&
                  near(r.bearing_deg, 30.0, 0.05),
              fmt("dist %.1f brg %.2f", std::hypot(e, n), r.bearing_deg));

        ot::TeleFrame nogps = s;
        nogps.has_gps = false;
        ot::RangeEstimator est6(cfg, W, H, cmap);
        check("no drone GPS -> no position", !est6.update(car, &nogps).has_geo);
    }

    std::printf("== map math ==\n");
    {
        double x, y;
        ot::latlon_to_world_px(0.0, 0.0, 0, x, y);
        check("world centre at zoom 0", near(x, 128.0, 1e-9) && near(y, 128.0, 1e-9));
        ot::latlon_to_world_px(0.0, 180.0, 3, x, y);
        check("east edge at zoom 3", near(x, 2048.0, 1e-9));
        ot::latlon_to_world_px(60.0, 0.0, 1, x, y);
        check("north is up (smaller y)", y < 256.0);
        {
            double wx, wy, lat, lon;
            ot::latlon_to_world_px(52.150959, 21.094030, 17, wx, wy);
            ot::world_px_to_latlon(wx, wy, 17, lat, lon);
            check("world pixels convert back to the same lat/lon",
                  near(lat, 52.150959, 1e-9) && near(lon, 21.094030, 1e-9));
        }
        check("metres per pixel at the equator", near(ot::meters_per_px(0.0, 0), 156543.03, 0.01));

        check("single point fits at the closest zoom",
              ot::fit_zoom(52.0, 52.0, 21.0, 21.0, 480, 720, 0.6, 3, 19) == 19);
        // 1 km wide at 52 N in a 480 px panel at 60 % fill needs >= 3.47 m/px:
        // zoom 14 (5.88 m/px) fits, zoom 15 (2.94) does not.
        double lat1, lon1;
        ot::destination(52.0, 21.0, 90.0, 1000.0, lat1, lon1);
        check("1 km box -> zoom 14", ot::fit_zoom(52.0, 52.0, 21.0, lon1, 480, 720, 0.6, 3, 19) == 14);
        check("fit_zoom respects min_zoom",
              ot::fit_zoom(-60.0, 60.0, -170.0, 170.0, 480, 720, 0.6, 3, 19) == 3);

        check("nice scale lengths", near(ot::nice_scale_m(120.0), 100.0, 1e-9) &&
                                        near(ot::nice_scale_m(49.0), 20.0, 1e-9) &&
                                        near(ot::nice_scale_m(5.0), 5.0, 1e-9) &&
                                        near(ot::nice_scale_m(0.7), 0.5, 1e-9) &&
                                        near(ot::nice_scale_m(0.0), 0.0, 1e-9));
        check("field of view", near(ot::hfov_deg(36.0), 53.13, 0.01) && near(ot::hfov_deg(0.0), 0.0, 1e-9));
    }

    std::printf("== tile cache (offline, disk) ==\n");
    {
        ot::MapCfg mc;
        mc.offline = true;
        mc.cache_dir = temp_dir("ot_test_tiles");
        mc.tile_url = "https://tiles.invalid/{z}/{x}/{y}.png";
        ot::TileCache cache(mc);
        check("url template", cache.url_for(17, 73254, 43019) == "https://tiles.invalid/17/73254/43019.png",
              cache.url_for(17, 73254, 43019));
        check("out-of-range tiles are refused", cache.get(3, 8, 0).empty() && cache.get(-1, 0, 0).empty());

        // Plant a tile on disk where the cache expects it, then ask for it.
        const std::string dir = cache.cache_dir() + "/5/3";
        std::filesystem::create_directories(dir);
        cv::Mat tile(256, 256, CV_8UC3, cv::Scalar(10, 200, 30));
        cv::imwrite(dir + "/7.png", tile);
        std::filesystem::rename(dir + "/7.png", dir + "/7.tile");
        check("first get() never blocks (returns empty, queues)", cache.get(5, 3, 7).empty());
        cv::Mat got;
        for (int i = 0; i < 200 && got.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            got = cache.get(5, 3, 7);
        }
        check("tile is loaded from disk in the background",
              !got.empty() && got.cols == 256 && got.at<cv::Vec3b>(100, 100)[1] == 200);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        check("a missing tile stays missing offline", cache.get(5, 3, 8).empty() && !cache.network_down());
    }

    std::printf("== map panel (offline render) ==\n");
    {
        ot::MapCfg mc;
        mc.offline = true;
        mc.cache_dir = temp_dir("ot_test_tiles_panel");
        ot::MapPanel panel(mc, 480, 720);

        ot::TargetState none;
        cv::Mat img = panel.render(nullptr, none, false, 0, "");
        check("renders without telemetry", img.cols == 480 && img.rows == 720 && img.type() == CV_8UC3);

        ot::TeleFrame d = tele(52.150959, 21.094030, 64.0, 70.0);
        d.heading = -71.0;
        d.has_heading = true;
        img = panel.render(&d, none, false, 1, "");
        check("drone only -> default zoom", panel.zoom() == mc.default_zoom && panel.target_trail_size() == 0);

        // A locked target 150 m out, driving east 1 m per inference frame.
        int64_t frame = 2;
        for (int i = 0; i < 120; ++i, ++frame) {
            ot::TargetState t = target_at(960, 540);
            t.frame_index = frame;
            t.has_geo = true;
            t.range_ground_m = 150.0f;
            ot::destination(d.lat, d.lon, -71.0, 150.0, t.target_lat, t.target_lon);
            double lat2, lon2;
            ot::destination(t.target_lat, t.target_lon, 90.0, 1.0 * i, lat2, lon2);
            t.target_lat = lat2;
            t.target_lon = lon2;
            img = panel.render(&d, t, true, frame, "car");
            img = panel.render(&d, t, true, frame, "car");   // same latched result twice
        }
        check("trail grows as the target moves", panel.target_trail_size() > 100,
              fmt("%.0f points", static_cast<double>(panel.target_trail_size())));
        check("trail is not double-counted for a repeated result", panel.target_trail_size() < 260);
        check("view zooms to frame drone + target", panel.zoom() >= 16 && panel.zoom() <= 18,
              fmt("zoom %.0f", panel.zoom()));
        if (const char* out = std::getenv("OT_TEST_MAP_PNG")) {
            cv::imwrite(out, img);
            std::printf("  saved %s\n", out);
        }

        ot::TargetState other = target_at(960, 540, 2, /*lock_id=*/2);
        other.frame_index = frame;
        other.has_geo = true;
        ot::destination(d.lat, d.lon, 20.0, 80.0, other.target_lat, other.target_lon);
        panel.render(&d, other, true, frame++, "car");
        check("a new lock starts a new trail", panel.target_trail_size() == 1);

        // Distance but no heading: no position, the panel shows a ring instead.
        ot::TeleFrame nh = d;
        nh.has_heading = false;
        ot::TargetState ring = target_at(960, 540, 2, 3);
        ring.frame_index = frame;
        ring.range_ground_m = 300.0f;
        img = panel.render(&nh, ring, true, frame++, "car");
        check("no heading -> no trail, view makes room for the ring",
              panel.target_trail_size() == 0 && panel.zoom() <= 16, fmt("zoom %.0f", panel.zoom()));

        const int zoom_locked = panel.zoom();
        panel.render(&d, none, false, frame++, "");
        check("reset clears the trail but keeps the zoom (no jump at unlock)",
              panel.target_trail_size() == 0 && panel.zoom() == zoom_locked,
              fmt("zoom %.0f", panel.zoom()));

        // Manual view. Events as highgui sends them (GTK convention is detected
        // from the build, so drive the zoom through the keyboard entry point).
        for (int i = 0; i < 80; ++i) panel.render(&d, none, false, frame++, "");   // let the view settle
        check("follows until the user takes over", panel.following() && !panel.needs_redraw());
        panel.zoom_by(-1);
        check("zoom out by one level", panel.zoom() == zoom_locked - 1 && !panel.following());
        check("a view change asks for a redraw", panel.needs_redraw());
        img = panel.redraw();
        check("redraw works between frames", img.cols == 480 && img.rows == 720 && !panel.needs_redraw());
        panel.render(&d, none, false, frame++, "");
        check("a manual zoom survives new frames", panel.zoom() == zoom_locked - 1);
        for (int i = 0; i < 40; ++i) panel.zoom_by(+1);
        check("zoom stops at the configured limit", panel.zoom() == mc.max_zoom);

        // Drag 100 px to the right: the map moves with the cursor, so the drone
        // (which was at the centre) ends up 100 px right of the centre.
        panel.follow();
        for (int i = 0; i < 60; ++i) panel.render(&d, none, false, frame++, "");   // settle
        const auto blue_x = [&](const cv::Mat& m) {      // mean x of the drone marker fill
            double sx = 0; int n = 0;
            for (int y = 0; y < m.rows; ++y)
                for (int x = 0; x < m.cols; ++x) {
                    const cv::Vec3b& p = m.at<cv::Vec3b>(y, x);
                    if (p[0] == 235 && p[1] == 110 && p[2] == 20) { sx += x; ++n; }
                }
            return n ? sx / n : -1.0;
        };
        ot::TeleFrame nohead = d;                        // no wedge: only the marker is blue
        nohead.has_heading = false;
        img = panel.render(&nohead, none, false, frame++, "");
        const double x_before = blue_x(img);
        panel.on_mouse(cv::EVENT_LBUTTONDOWN, 200, 300, 0);
        panel.on_mouse(cv::EVENT_MOUSEMOVE, 250, 300, 0);
        panel.on_mouse(cv::EVENT_MOUSEMOVE, 300, 300, 0);
        panel.on_mouse(cv::EVENT_LBUTTONUP, 300, 300, 0);
        panel.on_mouse(cv::EVENT_MOUSEMOVE, 400, 300, 0);   // button is up: must not pan
        img = panel.render(&nohead, none, false, frame++, "");
        const double x_after = blue_x(img);
        check("dragging pans the map with the cursor",
              !panel.following() && std::fabs((x_after - x_before) - 100.0) < 2.0,
              fmt("drone moved %.1f px", x_after - x_before));
        for (int i = 0; i < 40; ++i) img = panel.render(&nohead, none, false, frame++, "");
        check("a panned view stays put (does not glide back)",
              std::fabs(blue_x(img) - x_after) < 1.0, fmt("drone at x=%.1f", blue_x(img)));
        panel.on_mouse(cv::EVENT_LBUTTONDOWN, 240, 300, 0);
        panel.on_mouse(cv::EVENT_LBUTTONUP, 240, 300, 0);
        check("a plain click does not move the map", blue_x(panel.redraw()) == blue_x(img));
        panel.on_mouse(cv::EVENT_LBUTTONDBLCLK, 240, 300, 0);
        check("double-click hands the view back", panel.following());
        for (int i = 0; i < 80; ++i) img = panel.render(&nohead, none, false, frame++, "");
        check("following re-centres on the drone", std::fabs(blue_x(img) - 240.0) < 3.0,
              fmt("drone at x=%.1f", blue_x(img)));
        panel.on_mouse(cv::EVENT_LBUTTONDOWN, -50, 300, 0);   // a click on the video
        panel.on_mouse(cv::EVENT_MOUSEMOVE, 100, 300, 0);
        check("a drag that starts outside the panel is ignored", panel.following());
        const size_t before = panel.drone_trail_size();
        ot::TeleFrame moved = d;
        ot::destination(d.lat, d.lon, 0.0, 30.0, moved.lat, moved.lon);
        panel.render(&moved, none, false, frame++, "");
        check("drone track grows when the drone moves", panel.drone_trail_size() == before + 1);
        panel.render(&moved, none, false, 0, "");
        check("video restart clears the tracks", panel.drone_trail_size() == 1);
    }

    if (const char* out = std::getenv("OT_TEST_MAP_ONLINE")) {
        std::printf("== map panel (online smoke, downloads tiles) ==\n");
        ot::MapCfg mc;
        mc.cache_dir = temp_dir("ot_test_tiles_online");
        ot::MapPanel panel(mc, 480, 720);
        ot::TeleFrame d = tele(51.477928, -0.001545, 60.0, 70.0);   // a public landmark
        d.heading = 40.0;
        d.has_heading = true;
        cv::Mat img;
        int64_t frame = 0;
        const auto t0 = std::chrono::steady_clock::now();
        // Keep rendering while tiles arrive (the loader is asynchronous).
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(8)) {
            ot::TargetState t = target_at(960, 540);
            t.frame_index = frame;
            t.has_geo = true;
            t.range_ground_m = 120.0f;
            ot::destination(d.lat, d.lon, 40.0 + 0.2 * frame, 120.0, t.target_lat, t.target_lon);
            img = panel.render(&d, t, true, frame++, "car");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        // A map tile is colourful; the offline placeholder is flat dark grey.
        cv::Scalar mean, dev;
        cv::meanStdDev(img(cv::Rect(0, 200, 480, 300)), mean, dev);
        check("tiles were downloaded and drawn", dev[0] + dev[1] + dev[2] > 30.0,
              fmt("pixel spread %.1f", dev[0] + dev[1] + dev[2]));
        cv::imwrite(out, img);
        std::printf("  saved %s\n", out);
    }

    if (argc > 1) {
        std::printf("== embedded metadata track of: %s ==\n", argv[1]);
        std::vector<ot::TeleFrame> recs;
        const std::string side = ot::find_telemetry_sidecar(argv[1]);
        if (!side.empty()) {
            try { recs = ot::parse_telemetry(side); } catch (const std::exception&) {}
        }
        const size_t side_n = recs.size();
        const auto emb = ot::parse_embedded_telemetry(argv[1]);
        check("video has a readable metadata track", !emb.empty(),
              fmt("%.0f records (telemetry file: %.0f)", static_cast<double>(emb.size()),
                  static_cast<double>(side_n)));
        if (!emb.empty()) {
            if (side_n > 0) {
                const auto& a = recs.front();
                const auto& b = emb.front();
                check("embedded GPS/altitude agree with the telemetry file",
                      near(a.lat, b.lat, 2e-6) && near(a.lon, b.lon, 2e-6) && near(a.rel_alt, b.rel_alt, 0.01),
                      fmt("dlat %.7f dlon %.7f dalt %.3f", a.lat - b.lat, a.lon - b.lon, a.rel_alt - b.rel_alt));
            }
            ot::merge_telemetry(recs, emb);
            const ot::HeadingInfo hi = ot::resolve_heading(recs, kNaN);
            check("camera heading resolved", hi.frames > 0,
                  fmt("offset %+.2f deg on %.0f records", hi.offset_deg, static_cast<double>(hi.frames)));
            const auto& f0 = recs.front();
            const auto& fN = recs.back();
            std::printf("  first: gps %.6f,%.6f alt %.1f  body yaw %.1f  gimbal p/r/y %.1f/%.1f/%.1f  heading %.1f\n",
                        f0.lat, f0.lon, f0.rel_alt, f0.body_yaw, f0.gb_pitch, f0.gb_roll, f0.gb_yaw, f0.heading);
            std::printf("  last:  gps %.6f,%.6f alt %.1f  body yaw %.1f  gimbal p/r/y %.1f/%.1f/%.1f  heading %.1f\n",
                        fN.lat, fN.lon, fN.rel_alt, fN.body_yaw, fN.gb_pitch, fN.gb_roll, fN.gb_yaw, fN.heading);
        }
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
