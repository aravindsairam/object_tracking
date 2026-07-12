#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ot {

// One DJI SRT telemetry record (~one per video frame). Values the parser could
// not find stay 0 with their has_* flag false; frame is the 1-based FrameCnt.
struct TeleFrame {
    int         frame = 0;
    std::string time;               // wall-clock "YYYY-MM-DD HH:MM:SS.mmm"
    double      lat = 0, lon = 0;   // drone GPS [deg]
    double      rel_alt = 0;        // barometric altitude above takeoff [m]
    double      abs_alt = 0;
    double      focal_len = 0;      // 35mm-EQUIVALENT focal [mm]; changes with zoom
    double      gb_yaw = 0;         // gimbal attitude [deg]; only enterprise DJI
    double      gb_pitch = 0;       // SRTs carry these (0 level, -90 nadir) —
    double      gb_roll = 0;        // Mavic Fly-app SRTs do not.
    bool        has_gps = false;
    bool        has_focal = false;
    bool        has_attitude = false;   // gb_pitch was present in the record
};

// Parses a DJI SRT sidecar. Throws std::runtime_error if no records parse.
std::vector<TeleFrame> parse_srt(const std::string& path);

// Returns the path of an existing "<video stem>.SRT" / ".srt" next to the video,
// or "" if none (also "" for non-file sources like RTSP/camera indices).
std::string find_srt_sidecar(const std::string& video_path);

// Per-frame telemetry lookup, read-only after construction (safe to share
// across the inference and display threads without locking).
class Telemetry {
public:
    explicit Telemetry(std::vector<TeleFrame> records) : recs_(std::move(records)) {}

    // Record for a 0-based decode frame index (SRT FrameCnt is 1-based, so
    // recs_[index] is the matching record); clamped past the end, nullptr if
    // empty or index < 0.
    const TeleFrame* at(int64_t frame_index) const {
        if (recs_.empty() || frame_index < 0) return nullptr;
        const size_t i = static_cast<size_t>(frame_index);
        return &recs_[i < recs_.size() ? i : recs_.size() - 1];
    }

    size_t size() const { return recs_.size(); }
    bool   has_any_attitude() const {
        for (const TeleFrame& f : recs_)
            if (f.has_attitude) return true;
        return false;
    }

private:
    std::vector<TeleFrame> recs_;
};

}  // namespace ot
