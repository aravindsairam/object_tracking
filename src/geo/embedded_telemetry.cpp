// Embedded camera-metadata track: per-frame airframe heading + gimbal attitude
// stored inside the video file, next to the picture. The sidecar telemetry text
// file is a reduced copy of this track (GPS / altitude / focal only), so this is
// where camera heading and pitch come from on cameras whose sidecar lacks them.
//
// The track is a data stream with one length-delimited protobuf message per
// video frame. There is no published schema; the field paths below were worked
// out against real footage (GPS and altitudes match the sidecar exactly, the
// airframe yaw matches the GPS course in straight flight). Only the handful of
// fields we use are decoded, with a schema-less reader, and every value is
// range-checked, so an unknown layout simply yields no telemetry.
#include "ot/geodesy.hpp"
#include "ot/telemetry.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace ot {

namespace {
// ---- minimal schema-less protobuf reader -----------------------------------
enum : int { kVarint = 0, kFixed64 = 1, kLen = 2, kFixed32 = 5 };

struct PbField {
    uint32_t       num = 0;
    int            wire = -1;
    uint64_t       value = 0;       // kVarint
    const uint8_t* data = nullptr;  // kLen / kFixed32 / kFixed64 payload
    size_t         len = 0;
};

// Iterates the fields of one message. Any malformed byte ends the iteration
// (next() returns false), so callers never read out of bounds.
class PbReader {
public:
    PbReader(const uint8_t* d, size_t n) : p_(d), end_(d ? d + n : d) {}

    bool next(PbField& f) {
        if (!p_ || p_ >= end_) return false;
        uint64_t key = 0;
        if (!varint(key)) return fail();
        f = PbField{};
        f.num = static_cast<uint32_t>(key >> 3);
        f.wire = static_cast<int>(key & 7);
        switch (f.wire) {
            case kVarint:
                if (!varint(f.value)) return fail();
                return true;
            case kFixed64: return take(8, f);
            case kFixed32: return take(4, f);
            case kLen: {
                uint64_t n = 0;
                if (!varint(n) || n > static_cast<uint64_t>(end_ - p_)) return fail();
                return take(static_cast<size_t>(n), f);
            }
            default: return fail();   // groups / unknown wire types: stop
        }
    }

private:
    bool varint(uint64_t& v) {
        v = 0;
        for (int shift = 0; shift < 64 && p_ < end_; shift += 7) {
            const uint8_t c = *p_++;
            v |= static_cast<uint64_t>(c & 0x7f) << shift;
            if (!(c & 0x80)) return true;
        }
        return false;
    }
    bool take(size_t n, PbField& f) {
        if (static_cast<size_t>(end_ - p_) < n) return fail();
        f.data = p_;
        f.len = n;
        p_ += n;
        return true;
    }
    bool fail() {
        p_ = end_;
        return false;
    }

    const uint8_t* p_;
    const uint8_t* end_;
};

// First field `num` with wire type `wire` inside the message `in`.
bool find_field(const PbField& in, uint32_t num, int wire, PbField& out) {
    PbReader r(in.data, in.len);
    PbField f;
    while (r.next(f)) {
        if (f.num == num && f.wire == wire) {
            out = f;
            return true;
        }
    }
    return false;
}

// An attitude triple: three signed varints in tenths of a degree, fields 1..3.
// proto3 omits zero values, so a missing field is a genuine 0. Returns false
// when any value is outside a plausible angle range (wrong layout).
bool read_tenths(const PbField& msg, double out_deg[3]) {
    out_deg[0] = out_deg[1] = out_deg[2] = 0.0;
    PbReader r(msg.data, msg.len);
    PbField f;
    while (r.next(f)) {
        if (f.wire != kVarint || f.num < 1 || f.num > 3) continue;
        const int64_t tenths = static_cast<int64_t>(f.value);
        if (tenths < -3600 || tenths > 3600) return false;
        out_deg[f.num - 1] = static_cast<double>(tenths) / 10.0;
    }
    return true;
}

double read_f64(const PbField& f) {
    double v = 0.0;
    if (f.len == 8) std::memcpy(&v, f.data, 8);
    return v;
}
float read_f32(const PbField& f) {
    float v = 0.0f;
    if (f.len == 4) std::memcpy(&v, f.data, 4);
    return v;
}
}  // namespace

bool decode_embedded_packet(const uint8_t* data, size_t size, TeleFrame& out) {
    if (!data || size == 0) return false;
    PbField top;
    top.data = data;
    top.len = size;

    // Layout:  3 { 3 { airframe }  4 { gimbal } }
    //   airframe: 3 { pitch, roll, yaw } [0.1 deg]
    //             4 { 1 { 2: lat f64, 3: lon f64 }  2: abs altitude [mm] }
    //             5 { 1: altitude above takeoff [mm] f32 }
    //   gimbal:   3 { pitch, roll, yaw } [0.1 deg]
    PbField root;
    if (!find_field(top, 3, kLen, root)) return false;

    bool any = false;
    PbField sec, m, sub;
    if (find_field(root, 3, kLen, sec)) {
        double a[3];
        if (find_field(sec, 3, kLen, m) && read_tenths(m, a)) {
            out.body_yaw = wrap180(a[2]);
            out.has_body_yaw = true;
            any = true;
        }
        if (find_field(sec, 4, kLen, m)) {
            PbField flat, flon;
            if (find_field(m, 1, kLen, sub) && find_field(sub, 2, kFixed64, flat) &&
                find_field(sub, 3, kFixed64, flon)) {
                const double lat = read_f64(flat), lon = read_f64(flon);
                const bool sane = std::isfinite(lat) && std::isfinite(lon) &&
                                  std::fabs(lat) <= 90.0 && std::fabs(lon) <= 180.0 &&
                                  !(lat == 0.0 && lon == 0.0);
                if (sane) {
                    out.lat = lat;
                    out.lon = lon;
                    out.has_gps = true;
                    any = true;
                }
            }
            if (find_field(m, 2, kVarint, sub))
                out.abs_alt = static_cast<double>(static_cast<int64_t>(sub.value)) / 1000.0;
        }
        if (find_field(sec, 5, kLen, m) && find_field(m, 1, kFixed32, sub)) {
            const double rel = static_cast<double>(read_f32(sub)) / 1000.0;
            if (std::isfinite(rel) && std::fabs(rel) < 20000.0) out.rel_alt = rel;
        }
    }
    if (find_field(root, 4, kLen, sec)) {
        double g[3];
        if (find_field(sec, 3, kLen, m) && read_tenths(m, g)) {
            out.gb_pitch = g[0];
            out.gb_roll = g[1];
            out.gb_yaw = wrap180(g[2]);
            out.has_attitude = true;
            out.has_gb_yaw = true;
            any = true;
        }
    }
    return any;
}

std::vector<TeleFrame> parse_embedded_telemetry(const std::string& video_path) {
    std::vector<TeleFrame> recs;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(video_path, ec)) return recs;

    // The decoder opens this same file and already reports anything odd about
    // it; keep this second, metadata-only open quiet.
    const int log_level = av_log_get_level();
    av_log_set_level(AV_LOG_ERROR);
    AVFormatContext* fmt = nullptr;
    const int opened = avformat_open_input(&fmt, video_path.c_str(), nullptr, nullptr);
    av_log_set_level(log_level);
    if (opened < 0) return recs;

    // The metadata track is a data stream; pick it by its handler name.
    int meta = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        const AVStream* st = fmt->streams[i];
        if (st->codecpar->codec_type != AVMEDIA_TYPE_DATA) continue;
        const AVDictionaryEntry* h = av_dict_get(st->metadata, "handler_name", nullptr, 0);
        if (h && h->value && std::strstr(h->value, "CAM meta")) {
            meta = static_cast<int>(i);
            break;
        }
    }
    if (meta < 0) {
        avformat_close_input(&fmt);
        return recs;
    }
    // Discard everything else so the demuxer never reads the video samples:
    // a multi-GB clip costs only its few hundred KB of metadata.
    for (unsigned i = 0; i < fmt->nb_streams; ++i)
        fmt->streams[i]->discard = static_cast<int>(i) == meta ? AVDISCARD_DEFAULT
                                                               : AVDISCARD_ALL;

    AVPacket* pkt = av_packet_alloc();
    size_t decoded = 0;
    while (pkt && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == meta) {
            // One packet per video frame, in order: keep a record even when a
            // packet does not decode so later frames stay aligned.
            TeleFrame f;
            if (decode_embedded_packet(pkt->data, static_cast<size_t>(pkt->size), f)) ++decoded;
            f.frame = static_cast<int>(recs.size()) + 1;
            recs.push_back(std::move(f));
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    avformat_close_input(&fmt);

    if (decoded == 0) recs.clear();   // a track we cannot read is no telemetry
    return recs;
}

void merge_telemetry(std::vector<TeleFrame>& base, const std::vector<TeleFrame>& extra) {
    if (extra.empty()) return;
    if (base.empty()) {
        base = extra;
        return;
    }
    const size_t n = std::min(base.size(), extra.size());
    for (size_t i = 0; i < n; ++i) {
        TeleFrame& b = base[i];
        const TeleFrame& e = extra[i];
        if (!b.has_gps && e.has_gps) {
            b.lat = e.lat;
            b.lon = e.lon;
            b.has_gps = true;
        }
        if (!b.has_attitude && e.has_attitude) {
            b.gb_pitch = e.gb_pitch;
            b.gb_roll = e.gb_roll;
            b.has_attitude = true;
        }
        if (!b.has_gb_yaw && e.has_gb_yaw) {
            b.gb_yaw = e.gb_yaw;
            b.has_gb_yaw = true;
        }
        if (!b.has_body_yaw && e.has_body_yaw) {
            b.body_yaw = e.body_yaw;
            b.has_body_yaw = true;
        }
    }
}

HeadingInfo resolve_heading(std::vector<TeleFrame>& records, double offset_override_deg) {
    HeadingInfo info;

    // Gap between the two yaw readings wherever a record has both.
    std::vector<double> gaps;
    bool any_gimbal = false;
    for (const TeleFrame& f : records) {
        any_gimbal = any_gimbal || f.has_gb_yaw;
        if (f.has_gb_yaw && f.has_body_yaw) gaps.push_back(wrap180(f.body_yaw - f.gb_yaw));
    }

    double offset = 0.0;
    if (std::isfinite(offset_override_deg)) {
        offset = offset_override_deg;
    } else if (!gaps.empty()) {
        // Robust circular mean: the plain circular mean puts us on the right
        // branch of the wrap, the median of the residuals around it shrugs off
        // the transient gaps while the airframe turns under a lagging gimbal.
        double s = 0.0, c = 0.0;
        for (const double g : gaps) {
            s += std::sin(g * kDegToRad);
            c += std::cos(g * kDegToRad);
        }
        const double mean = std::atan2(s, c) * kRadToDeg;
        std::vector<double> res;
        res.reserve(gaps.size());
        for (const double g : gaps) res.push_back(wrap180(g - mean));
        const auto mid = res.begin() + res.size() / 2;
        std::nth_element(res.begin(), mid, res.end());
        offset = wrap180(mean + *mid);
        info.offset_auto = true;
    }

    for (TeleFrame& f : records) {
        if (f.has_gb_yaw) {
            f.heading = wrap180(f.gb_yaw + offset);
            f.has_heading = true;
        } else if (f.has_body_yaw) {
            f.heading = wrap180(f.body_yaw);
            f.has_heading = true;
        } else {
            f.has_heading = false;
        }
        if (f.has_heading) ++info.frames;
    }

    if (info.frames == 0) {
        info.source = HeadingInfo::Source::None;
    } else if (!any_gimbal) {
        info.source = HeadingInfo::Source::BodyYaw;
    } else if (info.offset_auto || std::isfinite(offset_override_deg)) {
        info.source = HeadingInfo::Source::GimbalPlusOffset;
        info.offset_deg = offset;
    } else {
        info.source = HeadingInfo::Source::GimbalYaw;   // no reference: taken as north-referenced
    }
    return info;
}

}  // namespace ot
