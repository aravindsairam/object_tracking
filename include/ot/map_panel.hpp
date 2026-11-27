#pragma once

#include "ot/config.hpp"
#include "ot/target_state.hpp"
#include "ot/telemetry.hpp"
#include "ot/tile_cache.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ot {

// ---- pure helpers (unit-tested in test_geo_map) -----------------------------

// Closest zoom in [min_zoom, max_zoom] at which the lat/lon box still fits
// inside `fill` (0..1) of a panel_w x panel_h panel. A degenerate (single
// point) box fits everywhere and returns max_zoom.
int fit_zoom(double lat_min, double lat_max, double lon_min, double lon_max,
             int panel_w, int panel_h, double fill, int min_zoom, int max_zoom);

// Largest "round" length (1, 2 or 5 x 10^n metres) not exceeding max_m.
double nice_scale_m(double max_m);

// Horizontal field of view [deg] of a 35mm-equivalent focal length.
double hfov_deg(double equiv_focal_mm);

// Map side panel: a north-up street map with the drone on it and, while a
// target is locked, the target's estimated position and the trail it has left.
//
//   nothing locked            drone marker + its own track + camera view wedge
//   locked, position known    + target marker, target trail, drone->target line
//   locked, no camera heading   a ring around the drone at the estimated ground
//                               distance ("somewhere on this circle")
//
// The view follows the action: it frames drone + target + trail and zooms out
// when they no longer fit. It zooms back in only when they would fit with room
// to spare, and never just because a lock ended, so locking and releasing
// targets does not make the map jump between zoom levels.
//
// The mouse takes over when wanted: drag pans, the wheel zooms, a double-click
// hands the view back to following. A manual zoom keeps following the drone;
// a manual pan stays where it was put.
//
// Tiles come from a TileCache, which never blocks: a tile that is not loaded
// yet is drawn from coarser or finer tiles already in memory until it arrives.
//
// Owns its trails, so call render() once per displayed frame. Everything here
// (including on_mouse) belongs to the display thread.
class MapPanel {
public:
    MapPanel(const MapCfg& cfg, int width, int height);

    // tele:   telemetry of the DISPLAYED frame (drone GPS / heading), or null.
    // target: latest lock result; `locked` as in the main loop (true through
    //         Locked / Coasting / Lost, false once reset).
    // label:  short name for the target (its class), shown next to the marker.
    cv::Mat render(const TeleFrame* tele, const TargetState& target, bool locked,
                   int64_t frame_index, const std::string& label);

    // Mouse event in PANEL pixel coordinates (they may lie outside the panel;
    // such events only end a drag). `event` / `flags` are highgui's.
    void on_mouse(int event, int x, int y, int flags);
    void zoom_by(int steps);        // keyboard: +1 = in, -1 = out, about the centre
    void follow();                  // drop the manual pan / zoom

    // The panel is normally redrawn with every video frame. Between frames
    // (paused video) needs_redraw() says the picture is stale - the view was
    // moved or a tile arrived - and redraw() draws it again from the inputs of
    // the last render(). Empty Mat before the first render().
    bool    needs_redraw() const;
    cv::Mat redraw();

    bool   following() const { return !manual_center_ && !manual_zoom_; }
    int    zoom() const { return zoom_; }
    size_t target_trail_size() const { return target_trail_.size(); }
    size_t drone_trail_size() const { return drone_trail_.size(); }

private:
    struct LL {
        double lat = 0, lon = 0;
    };
    void update_state(const TeleFrame* tele, const TargetState& target, bool locked,
                      int64_t frame_index);
    void update_view(const TargetState& target, bool locked);
    cv::Mat draw();
    void draw_tiles(cv::Mat& panel, double ox, double oy);
    void zoom_at(int steps, int x, int y);

    MapCfg    cfg_;
    int       w_, h_;
    TileCache tiles_;

    // drone
    bool            have_drone_ = false;
    LL              drone_;
    std::vector<LL> drone_trail_;

    // target
    bool            have_target_ = false;
    LL              target_;                 // smoothed
    std::vector<LL> target_trail_;
    int             trail_lock_id_ = -1;
    int64_t         last_target_frame_ = -1;
    LL              tmin_, tmax_;            // trail bounding box

    // view
    int     zoom_ = -1;
    bool    have_center_ = false;
    LL      center_;
    int64_t last_frame_ = -1;
    bool    manual_center_ = false;          // the user panned
    bool    manual_zoom_ = false;            // the user zoomed
    bool    easing_ = false;                 // centre still gliding to its goal

    // mouse
    bool      wheel_up_negative_ = false;    // backend convention, see map_panel.cpp
    bool      dragging_ = false;
    bool      drag_moved_ = false;
    cv::Point drag_last_;

    // inputs of the last render(), for redraw()
    bool        have_last_ = false;
    bool        last_has_tele_ = false;
    TeleFrame   last_tele_;
    TargetState last_target_;
    bool        last_locked_ = false;
    std::string last_label_;
    bool        dirty_ = false;              // view changed since the last draw
    uint64_t    seen_loaded_ = 0;            // tiles_.loaded() at the last draw
};

}  // namespace ot
