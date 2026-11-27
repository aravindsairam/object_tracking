#pragma once

#include <algorithm>
#include <cmath>

// Small-area geodesy + Web-Mercator helpers shared by the range estimator
// (target position) and the map panel (drawing). Header-only, pure math.
namespace ot {

constexpr double kEarthRadiusM = 6378137.0;   // WGS-84 equatorial radius [m]
constexpr double kDegToRad = M_PI / 180.0;
constexpr double kRadToDeg = 180.0 / M_PI;

// Wraps an angle to (-180, 180] degrees.
inline double wrap180(double deg) {
    double a = std::fmod(deg + 180.0, 360.0);
    if (a <= 0.0) a += 360.0;
    return a - 180.0;
}

// Point reached from (lat, lon) after dist_m along a compass bearing (degrees
// clockwise from north). Equirectangular: fine for the < few-km ranges here.
inline void destination(double lat, double lon, double bearing_deg, double dist_m,
                        double& out_lat, double& out_lon) {
    const double b = bearing_deg * kDegToRad;
    out_lat = lat + (dist_m * std::cos(b) / kEarthRadiusM) * kRadToDeg;
    out_lon = lon + (dist_m * std::sin(b) /
                     (kEarthRadiusM * std::cos(lat * kDegToRad))) * kRadToDeg;
}

// East/north offset [m] of (lat, lon) from (lat0, lon0).
inline void enu_offset(double lat0, double lon0, double lat, double lon,
                       double& east_m, double& north_m) {
    east_m  = (lon - lon0) * kDegToRad * kEarthRadiusM * std::cos(lat0 * kDegToRad);
    north_m = (lat - lat0) * kDegToRad * kEarthRadiusM;
}

// Web-Mercator "world pixel" coordinates at a zoom level, for 256 px tiles:
// x grows east, y grows SOUTH, the whole world is 256 * 2^zoom px square.
inline void latlon_to_world_px(double lat, double lon, int zoom, double& x, double& y) {
    const double n = 256.0 * std::ldexp(1.0, zoom);
    const double s = std::sin(std::max(-85.05112878, std::min(85.05112878, lat)) * kDegToRad);
    x = (lon + 180.0) / 360.0 * n;
    y = (0.5 - std::log((1.0 + s) / (1.0 - s)) / (4.0 * M_PI)) * n;
}

// Inverse of latlon_to_world_px.
inline void world_px_to_latlon(double x, double y, int zoom, double& lat, double& lon) {
    const double n = 256.0 * std::ldexp(1.0, zoom);
    lon = x / n * 360.0 - 180.0;
    lat = std::atan(std::sinh(M_PI * (1.0 - 2.0 * y / n))) * kRadToDeg;
}

// Ground metres covered by one map pixel at this latitude and zoom.
inline double meters_per_px(double lat, int zoom) {
    return 2.0 * M_PI * kEarthRadiusM * std::cos(lat * kDegToRad) /
           (256.0 * std::ldexp(1.0, zoom));
}

}  // namespace ot
