// Map side panel: drone position, camera view wedge, and the locked target's
// estimated position + trail on a north-up tile map. See map_panel.hpp.
#include "ot/map_panel.hpp"

#include "ot/geodesy.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ot {

namespace {
const cv::Scalar kBg(38, 38, 38);
const cv::Scalar kGrid(58, 58, 58);
const cv::Scalar kBlack(0, 0, 0);
const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kDrone(235, 110, 20);        // blue (BGR)
const cv::Scalar kWedge(255, 190, 110);
const cv::Scalar kTrail(205, 0, 205);         // magenta: reads on light and dark maps

constexpr double kTrailStepM      = 0.5;      // new target-trail point per this much travel
constexpr double kDroneStepM      = 1.0;
constexpr size_t kMaxTrailPoints  = 4000;
constexpr double kTargetSmoothing = 0.25;     // EMA weight of a new target fix
constexpr double kCenterSmoothing = 0.15;
constexpr double kFitFill         = 0.60;     // zoom out when the content outgrows this
constexpr double kZoomInFill      = 0.40;     // zoom in only when it would fit in this

cv::Scalar state_color(LockState s) {
    switch (s) {
        case LockState::Locked:   return {0, 200, 0};
        case LockState::Coasting: return {0, 170, 255};
        case LockState::Lost:     return {0, 0, 255};
        default:                  return {120, 120, 120};
    }
}

template <typename P>
double dist_m(const P& a, const P& b) {
    double e, n;
    enu_offset(a.lat, a.lon, b.lat, b.lon, e, n);
    return std::hypot(e, n);
}

// Text on a darkened backing box so it stays readable over any map. `org` is
// the text baseline-left; it is nudged to keep the whole label on the panel.
void label(cv::Mat& img, const std::string& text, cv::Point org, const cv::Scalar& color,
           double scale = 0.45) {
    int base = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &base);
    org.x = std::max(5, std::min(org.x, img.cols - ts.width - 5));
    org.y = std::max(ts.height + 5, std::min(org.y, img.rows - base - 5));
    const cv::Rect box = cv::Rect(org.x - 4, org.y - ts.height - 4, ts.width + 8,
                                  ts.height + base + 8) & cv::Rect(0, 0, img.cols, img.rows);
    if (box.area() > 0) {
        cv::Mat roi = img(box);
        roi.convertTo(roi, -1, 0.28);
    }
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

// A polyline with a dark casing under it.
template <typename P, typename ToPx>
void draw_trail(cv::Mat& img, const std::vector<P>& pts, const ToPx& to_px,
                const cv::Scalar& color, int thickness) {
    if (pts.size() < 2) return;
    std::vector<cv::Point> px;
    px.reserve(pts.size());
    for (const P& p : pts) px.push_back(to_px(p));
    cv::polylines(img, px, false, kBlack, thickness + 2, cv::LINE_AA);
    cv::polylines(img, px, false, color, thickness, cv::LINE_AA);
}

template <typename P>
void append_trail(std::vector<P>& trail, const P& p, double step_m) {
    if (!trail.empty() && dist_m(trail.back(), p) < step_m) return;
    trail.push_back(p);
    if (trail.size() > kMaxTrailPoints) {            // thin out: keep every other point
        size_t w = 0;
        for (size_t i = 0; i < trail.size(); i += 2) trail[w++] = trail[i];
        trail.resize(w);
    }
}

double to_compass(double deg) {
    const double a = wrap180(deg);
    return a < 0.0 ? a + 360.0 : a;
}

// Which way is "wheel up"? HighGUI's GTK backend reports it as a NEGATIVE
// delta, the Qt and Win32 backends as a positive one. Older OpenCV has no
// call that names the backend, so read it off the build description.
bool wheel_up_is_negative() {
    const std::string info = cv::getBuildInformation();
    for (const char* tag : {"GTK+:", "GUI:"}) {
        const size_t p = info.find(tag);
        if (p == std::string::npos) continue;
        const std::string line = info.substr(p, info.find('\n', p) - p);
        if (tag[1] == 'T' ? line.find("YES") != std::string::npos
                          : line.find("GTK") != std::string::npos)
            return true;
    }
    return false;
}
}  // namespace

int fit_zoom(double lat_min, double lat_max, double lon_min, double lon_max,
             int panel_w, int panel_h, double fill, int min_zoom, int max_zoom) {
    for (int z = max_zoom; z > min_zoom; --z) {
        double x0, y0, x1, y1;
        latlon_to_world_px(lat_max, lon_min, z, x0, y0);
        latlon_to_world_px(lat_min, lon_max, z, x1, y1);
        if (std::fabs(x1 - x0) <= fill * panel_w && std::fabs(y1 - y0) <= fill * panel_h)
            return z;
    }
    return min_zoom;
}

double nice_scale_m(double max_m) {
    if (!(max_m > 0.0)) return 0.0;
    const double p = std::pow(10.0, std::floor(std::log10(max_m)));
    for (const double m : {5.0, 2.0, 1.0})
        if (m * p <= max_m) return m * p;
    return p;
}

double hfov_deg(double equiv_focal_mm) {
    return equiv_focal_mm > 0.0 ? 2.0 * std::atan(18.0 / equiv_focal_mm) * kRadToDeg : 0.0;
}

MapPanel::MapPanel(const MapCfg& cfg, int width, int height)
    : cfg_(cfg), w_(std::max(64, width)), h_(std::max(64, height)), tiles_(cfg),
      wheel_up_negative_(wheel_up_is_negative()) {}

// ---- manual view -------------------------------------------------------------

void MapPanel::on_mouse(int event, int x, int y, int flags) {
    const bool inside = x >= 0 && y >= 0 && x < w_ && y < h_;
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            dragging_ = inside && have_center_ && zoom_ >= 0;
            drag_moved_ = false;
            drag_last_ = {x, y};
            break;
        case cv::EVENT_LBUTTONUP:
            dragging_ = false;
            break;
        case cv::EVENT_MOUSEMOVE: {
            if (!dragging_) break;
            // Leaving the panel ends the drag. (The button state in `flags` is
            // not reliable on every backend, and a release outside the window
            // is never reported, so the position is what we go by.)
            if (!inside) {
                dragging_ = false;
                break;
            }
            const int dx = x - drag_last_.x, dy = y - drag_last_.y;
            if (!drag_moved_ && std::abs(dx) + std::abs(dy) < 3) break;   // a click, not a drag
            drag_moved_ = true;
            drag_last_ = {x, y};
            double cx, cy;
            latlon_to_world_px(center_.lat, center_.lon, zoom_, cx, cy);
            world_px_to_latlon(cx - dx, cy - dy, zoom_, center_.lat, center_.lon);
            manual_center_ = true;
            dirty_ = true;
            break;
        }
        case cv::EVENT_LBUTTONDBLCLK:
            if (inside) follow();
            break;
        // Some backends report a vertical scroll as a horizontal-wheel event.
        case cv::EVENT_MOUSEWHEEL:
        case cv::EVENT_MOUSEHWHEEL: {
            if (!inside) break;
            const int d = cv::getMouseWheelDelta(flags);
            if (d == 0) break;
            zoom_at((wheel_up_negative_ ? d < 0 : d > 0) ? 1 : -1, x, y);
            break;
        }
        default:
            break;
    }
}

void MapPanel::zoom_at(int steps, int x, int y) {
    if (zoom_ < 0 || !have_center_) return;
    const int nz = std::max(cfg_.min_zoom, std::min(cfg_.max_zoom, zoom_ + steps));
    if (nz == zoom_) return;
    if (manual_center_) {
        // Keep the spot under the cursor where it is. (While the view still
        // follows the drone, the centre is not ours to move.)
        const double ax = x - w_ * 0.5, ay = y - h_ * 0.5;
        double cx, cy, lat, lon;
        latlon_to_world_px(center_.lat, center_.lon, zoom_, cx, cy);
        world_px_to_latlon(cx + ax, cy + ay, zoom_, lat, lon);
        latlon_to_world_px(lat, lon, nz, cx, cy);
        world_px_to_latlon(cx - ax, cy - ay, nz, center_.lat, center_.lon);
    }
    zoom_ = nz;
    manual_zoom_ = true;
    dirty_ = true;
}

void MapPanel::zoom_by(int steps) { zoom_at(steps, w_ / 2, h_ / 2); }

void MapPanel::follow() {
    if (following()) return;
    manual_center_ = manual_zoom_ = false;
    dragging_ = false;
    zoom_ = -1;          // fit again from scratch
    dirty_ = true;
}

bool MapPanel::needs_redraw() const {
    return have_last_ && (dirty_ || easing_ || tiles_.loaded() != seen_loaded_);
}

cv::Mat MapPanel::redraw() { return have_last_ ? draw() : cv::Mat(); }

void MapPanel::update_state(const TeleFrame* tele, const TargetState& target, bool locked,
                            int64_t frame_index) {
    // The video jumped backwards (loop restart / seek): the old trails belong
    // to a different pass over the clip.
    if (frame_index < last_frame_) {
        drone_trail_.clear();
        target_trail_.clear();
        have_target_ = false;
        last_target_frame_ = -1;
    }
    last_frame_ = frame_index;

    if (tele && tele->has_gps) {
        drone_ = {tele->lat, tele->lon};
        have_drone_ = true;
        append_trail(drone_trail_, drone_, kDroneStepM);
    }

    if (!locked) {                       // reset: nothing to show for the target
        target_trail_.clear();
        have_target_ = false;
        trail_lock_id_ = -1;
        last_target_frame_ = -1;
        return;
    }
    if (target.lock_id != trail_lock_id_) {   // a different target: start a new trail
        target_trail_.clear();
        have_target_ = false;
        trail_lock_id_ = target.lock_id;
        last_target_frame_ = -1;
    }
    // One fix per inference result: the display re-reads the same latched
    // result on several frames and must not smooth it in more than once.
    if (!target.valid() || !target.has_geo || target.frame_index == last_target_frame_) return;
    last_target_frame_ = target.frame_index;

    const LL fix{target.target_lat, target.target_lon};
    if (!have_target_) {
        target_ = fix;
        have_target_ = true;
    } else {
        target_.lat += kTargetSmoothing * (fix.lat - target_.lat);
        target_.lon += kTargetSmoothing * (fix.lon - target_.lon);
    }
    if (target_trail_.empty()) {
        tmin_ = tmax_ = target_;
    } else {
        tmin_ = {std::min(tmin_.lat, target_.lat), std::min(tmin_.lon, target_.lon)};
        tmax_ = {std::max(tmax_.lat, target_.lat), std::max(tmax_.lon, target_.lon)};
    }
    append_trail(target_trail_, target_, kTrailStepM);
}

void MapPanel::update_view(const TargetState& target, bool locked) {
    // Box around everything that should be on screen.
    LL lo = drone_, hi = drone_;
    bool multi = false;
    const auto extend = [&](const LL& p) {
        lo = {std::min(lo.lat, p.lat), std::min(lo.lon, p.lon)};
        hi = {std::max(hi.lat, p.lat), std::max(hi.lon, p.lon)};
        multi = true;
    };
    if (have_target_) {
        extend(target_);
        if (!target_trail_.empty()) {
            extend(tmin_);
            extend(tmax_);
        }
    } else if (locked && target.valid() && target.range_ground_m > 0.0f) {
        // No position, only a distance: make room for the ring around the drone.
        const double r = target.range_ground_m;
        const double dlat = r / kEarthRadiusM * kRadToDeg;
        const double dlon = dlat / std::max(0.01, std::cos(drone_.lat * kDegToRad));
        extend({drone_.lat - dlat, drone_.lon - dlon});
        extend({drone_.lat + dlat, drone_.lon + dlon});
    }

    if (manual_zoom_) {
        // the user's zoom stands
    } else if (!multi) {
        // Only the drone to show: any zoom frames it, so keep the current one.
        // (Going back to the default here made the map jump at every unlock.)
        if (zoom_ < 0) zoom_ = cfg_.default_zoom;
    } else {
        const int want = fit_zoom(lo.lat, hi.lat, lo.lon, hi.lon, w_, h_, kFitFill,
                                  cfg_.min_zoom, cfg_.max_zoom);
        if (zoom_ < 0 || want < zoom_) {
            zoom_ = want;                            // must zoom out to keep it in view
        } else if (want > zoom_) {
            const int relaxed = fit_zoom(lo.lat, hi.lat, lo.lon, hi.lon, w_, h_, kZoomInFill,
                                         cfg_.min_zoom, cfg_.max_zoom);
            if (relaxed > zoom_) zoom_ = relaxed;    // comfortably fits closer in
        }
    }

    easing_ = false;
    if (manual_center_ && have_center_) return;      // the user's centre stands

    const LL mid{(lo.lat + hi.lat) * 0.5, (lo.lon + hi.lon) * 0.5};
    bool snap = !have_center_;
    if (!snap) {
        // A jump bigger than the panel (new clip position) is not worth easing.
        const double jump_px = dist_m(center_, mid) / meters_per_px(mid.lat, zoom_);
        snap = jump_px > std::max(w_, h_);
    }
    if (snap) {
        center_ = mid;
        have_center_ = true;
    } else {
        center_.lat += kCenterSmoothing * (mid.lat - center_.lat);
        center_.lon += kCenterSmoothing * (mid.lon - center_.lon);
        easing_ = dist_m(center_, mid) / meters_per_px(mid.lat, zoom_) > 0.5;
    }
}

void MapPanel::draw_tiles(cv::Mat& panel, double ox, double oy) {
    const int n = 1 << zoom_;
    const int tx0 = static_cast<int>(std::floor(ox / 256.0));
    const int ty0 = static_cast<int>(std::floor(oy / 256.0));
    const int tx1 = static_cast<int>(std::floor((ox + w_ - 1) / 256.0));
    const int ty1 = static_cast<int>(std::floor((oy + h_ - 1) / 256.0));
    const cv::Rect bounds(0, 0, w_, h_);

    // Ask for the same area one level out first (the tiles on screen are
    // requested after it and so load before it): a zoom-out, which is what a
    // new lock usually causes, then has its tiles ready instead of popping in.
    if (zoom_ > cfg_.min_zoom) {
        const int pn = 1 << (zoom_ - 1);
        for (int ty = ty0 >> 1; ty <= ty1 >> 1; ++ty)
            for (int tx = tx0 >> 1; tx <= tx1 >> 1; ++tx)
                if (ty >= 0 && ty < pn) tiles_.get(zoom_ - 1, ((tx % pn) + pn) % pn, ty);
    }

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            const cv::Rect full(static_cast<int>(tx * 256.0 - ox),
                                static_cast<int>(ty * 256.0 - oy), 256, 256);
            const cv::Rect dst = full & bounds;
            if (dst.area() <= 0) continue;

            cv::Mat tile;
            if (ty >= 0 && ty < n) {
                const int txw = ((tx % n) + n) % n;   // the map wraps east-west
                tile = tiles_.get(zoom_, txw, ty);
                // Not loaded yet: stand in with what is already in memory, so a
                // zoom change is briefly soft instead of blank. After zooming
                // out that is the finer tiles shrunk together ...
                for (int down = 1; tile.empty() && down <= 2 && zoom_ + down <= 22; ++down) {
                    const int k = 1 << down, s = 256 >> down;
                    cv::Mat mosaic;
                    for (int j = 0; j < k; ++j) {
                        for (int i = 0; i < k; ++i) {
                            const cv::Mat child = tiles_.get(zoom_ + down, txw * k + i, ty * k + j, false);
                            if (child.empty()) continue;
                            if (mosaic.empty()) mosaic = cv::Mat(256, 256, CV_8UC3, kBg);
                            cv::resize(child, mosaic(cv::Rect(i * s, j * s, s, s)), {s, s}, 0, 0,
                                       cv::INTER_AREA);
                        }
                    }
                    tile = mosaic;
                }
                // ... and after zooming in, the matching corner of a coarser tile.
                for (int up = 1; tile.empty() && up <= 5 && zoom_ - up >= 0; ++up) {
                    const cv::Mat parent = tiles_.get(zoom_ - up, txw >> up, ty >> up, false);
                    if (parent.empty()) continue;
                    const int s = 256 >> up;
                    const int mask = (1 << up) - 1;
                    cv::resize(parent(cv::Rect((txw & mask) * s, (ty & mask) * s, s, s)), tile,
                               {256, 256}, 0, 0, cv::INTER_LINEAR);
                }
            }
            if (!tile.empty()) {
                tile(cv::Rect(dst.x - full.x, dst.y - full.y, dst.width, dst.height))
                    .copyTo(panel(dst));
            } else {
                cv::rectangle(panel, full, kGrid, 1);   // placeholder cell
            }
        }
    }
}

cv::Mat MapPanel::render(const TeleFrame* tele, const TargetState& target, bool locked,
                         int64_t frame_index, const std::string& name) {
    update_state(tele, target, locked, frame_index);
    last_has_tele_ = tele != nullptr;
    if (tele) last_tele_ = *tele;
    last_target_ = target;
    last_locked_ = locked;
    last_label_ = name;
    have_last_ = true;
    return draw();
}

cv::Mat MapPanel::draw() {
    const TeleFrame* tele = last_has_tele_ ? &last_tele_ : nullptr;
    const TargetState& target = last_target_;
    const bool locked = last_locked_;
    const std::string& name = last_label_;
    dirty_ = false;
    seen_loaded_ = tiles_.loaded();

    cv::Mat panel(h_, w_, CV_8UC3, kBg);
    if (!have_drone_) {
        label(panel, "MAP", {10, 22}, {220, 220, 220}, 0.6);
        const char* msg = "no drone position in telemetry";
        int base = 0;
        const cv::Size ts = cv::getTextSize(msg, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::putText(panel, msg, {(w_ - ts.width) / 2, h_ / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {150, 150, 150}, 1, cv::LINE_AA);
        return panel;
    }

    update_view(target, locked);

    // Panel origin in world pixels at this zoom (whole pixels: no tile seams).
    double cx, cy;
    latlon_to_world_px(center_.lat, center_.lon, zoom_, cx, cy);
    const double ox = std::floor(cx - w_ * 0.5), oy = std::floor(cy - h_ * 0.5);
    draw_tiles(panel, ox, oy);

    const auto to_px = [&](const LL& p) {
        double x, y;
        latlon_to_world_px(p.lat, p.lon, zoom_, x, y);
        return cv::Point(static_cast<int>(std::lround(x - ox)),
                         static_cast<int>(std::lround(y - oy)));
    };
    const double mpp = meters_per_px(center_.lat, zoom_);
    const cv::Point dp = to_px(drone_);
    const cv::Scalar tcol = state_color(target.state);
    const bool has_heading = tele && tele->has_heading;
    // Screen direction of a compass bearing: north is up, east is right.
    const auto dir = [](double bearing_deg) {
        const double b = bearing_deg * kDegToRad;
        return cv::Point2d(std::sin(b), -std::cos(b));
    };

    // ---- camera view wedge ---------------------------------------------------
    if (has_heading) {
        double half = hfov_deg(tele->has_focal ? tele->focal_len : 0.0) * 0.5;
        if (!(half > 0.0)) half = 20.0;
        half = std::max(3.0, std::min(60.0, half));
        double len = 80.0;
        if (have_target_) {
            const cv::Point tp = to_px(target_);
            len = std::max(60.0, std::min(2.0 * std::max(w_, h_),
                                          std::hypot(tp.x - dp.x, tp.y - dp.y) * 1.2));
        }
        std::vector<cv::Point> poly{dp};
        const int steps = 10;
        for (int i = 0; i <= steps; ++i) {
            const cv::Point2d d = dir(tele->heading - half + 2.0 * half * i / steps);
            poly.emplace_back(static_cast<int>(std::lround(dp.x + len * d.x)),
                              static_cast<int>(std::lround(dp.y + len * d.y)));
        }
        cv::Mat tint = panel.clone();
        cv::fillConvexPoly(tint, poly, kWedge, cv::LINE_AA);
        cv::addWeighted(tint, 0.35, panel, 0.65, 0.0, panel);
        cv::line(panel, dp, poly[1], kDrone, 1, cv::LINE_AA);
        cv::line(panel, dp, poly.back(), kDrone, 1, cv::LINE_AA);
    }

    // ---- drone track -----------------------------------------------------------
    draw_trail(panel, drone_trail_, to_px, kDrone, 1);

    // ---- target ------------------------------------------------------------------
    char buf[128];
    std::string target_line;
    if (have_target_) {
        const cv::Point tp = to_px(target_);
        cv::line(panel, dp, tp, kBlack, 3, cv::LINE_AA);
        cv::line(panel, dp, tp, tcol, 1, cv::LINE_AA);
        draw_trail(panel, target_trail_, to_px, kTrail, 2);
        cv::circle(panel, tp, 8, kBlack, cv::FILLED, cv::LINE_AA);
        cv::circle(panel, tp, 6, tcol, cv::FILLED, cv::LINE_AA);
        cv::circle(panel, tp, 2, kWhite, cv::FILLED, cv::LINE_AA);

        const double d = dist_m(drone_, target_);
        if (d < 995.0) std::snprintf(buf, sizeof(buf), "%s ~%.0fm", name.c_str(), d);
        else std::snprintf(buf, sizeof(buf), "%s ~%.1fkm", name.c_str(), d / 1000.0);
        label(panel, buf, {tp.x + 12, tp.y - 10}, kWhite);

        std::snprintf(buf, sizeof(buf), "target %.6f, %.6f%s", target_.lat, target_.lon,
                      target.state == LockState::Lost ? "  LOST" : "");
        target_line = buf;
    } else if (locked && target.valid() && target.range_ground_m > 0.0f) {
        // A distance but no direction: the target is somewhere on this ring.
        const int r = static_cast<int>(std::lround(target.range_ground_m / mpp));
        cv::circle(panel, dp, r, kBlack, 4, cv::LINE_AA);
        cv::circle(panel, dp, r, tcol, 2, cv::LINE_AA);
        std::snprintf(buf, sizeof(buf), "%s ~%.0fm, direction unknown", name.c_str(),
                      static_cast<double>(target.range_ground_m));
        label(panel, buf, {dp.x - 90, dp.y - r - 10}, kWhite);
        target_line = "target: no camera heading (ring = distance)";
    } else if (locked) {
        target_line = target.state == LockState::Lost ? "target: LOST"
                                                      : "target: no distance yet";
    }

    // ---- drone marker (on top) ----------------------------------------------------
    if (has_heading) {
        const cv::Point2d f = dir(tele->heading), s(-f.y, f.x);
        const std::vector<cv::Point> tri{
            {static_cast<int>(std::lround(dp.x + 13 * f.x)), static_cast<int>(std::lround(dp.y + 13 * f.y))},
            {static_cast<int>(std::lround(dp.x - 8 * f.x + 8 * s.x)), static_cast<int>(std::lround(dp.y - 8 * f.y + 8 * s.y))},
            {static_cast<int>(std::lround(dp.x - 8 * f.x - 8 * s.x)), static_cast<int>(std::lround(dp.y - 8 * f.y - 8 * s.y))}};
        cv::fillConvexPoly(panel, tri, kDrone, cv::LINE_AA);
        cv::polylines(panel, tri, true, kWhite, 2, cv::LINE_AA);
    } else {
        cv::circle(panel, dp, 8, kWhite, cv::FILLED, cv::LINE_AA);
        cv::circle(panel, dp, 6, kDrone, cv::FILLED, cv::LINE_AA);
    }

    // ---- text: header, north, scale, attribution --------------------------------
    label(panel, "MAP", {10, 22}, {230, 230, 230}, 0.55);
    if (has_heading)
        std::snprintf(buf, sizeof(buf), "drone %.6f, %.6f  hdg %.0f", drone_.lat, drone_.lon,
                      to_compass(tele->heading));
    else
        std::snprintf(buf, sizeof(buf), "drone %.6f, %.6f", drone_.lat, drone_.lon);
    label(panel, buf, {10, 46}, {230, 230, 230});
    if (!target_line.empty()) label(panel, target_line, {10, 68}, kWhite);
    if (!following())
        label(panel, "double-click: follow again", {10, target_line.empty() ? 68 : 90},
              {0, 220, 255}, 0.42);

    const cv::Point na(w_ - 22, 50);
    cv::arrowedLine(panel, na, {na.x, na.y - 30}, kBlack, 4, cv::LINE_AA, 0, 0.35);
    cv::arrowedLine(panel, na, {na.x, na.y - 30}, kWhite, 2, cv::LINE_AA, 0, 0.35);
    label(panel, "N", {na.x - 5, na.y + 20}, kWhite, 0.5);

    const double bar_m = nice_scale_m(mpp * std::min(140, w_ / 3));
    if (bar_m > 0.0) {
        const int len = static_cast<int>(std::lround(bar_m / mpp));
        const cv::Point a(12, h_ - 30), b(12 + len, h_ - 30);
        cv::line(panel, a, b, kBlack, 6);
        cv::line(panel, a, b, kWhite, 2);
        if (bar_m >= 1000.0) std::snprintf(buf, sizeof(buf), "%.0f km", bar_m / 1000.0);
        else std::snprintf(buf, sizeof(buf), "%.0f m", bar_m);
        label(panel, buf, {12, h_ - 40}, kWhite, 0.42);
    }
    if (tiles_.network_down())
        label(panel, "map tiles unavailable - showing cache", {12, h_ - 66}, {0, 200, 255}, 0.42);
    if (!cfg_.attribution.empty()) {
        int base = 0;
        const cv::Size ts = cv::getTextSize(cfg_.attribution, cv::FONT_HERSHEY_SIMPLEX, 0.36, 1, &base);
        label(panel, cfg_.attribution, {w_ - ts.width - 8, h_ - 8}, {235, 235, 235}, 0.36);
    }
    return panel;
}

}  // namespace ot
