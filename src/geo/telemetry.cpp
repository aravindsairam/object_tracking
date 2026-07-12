// DJI SRT sidecar parsing. Each subtitle block carries one frame's telemetry as
// "[key: value]" fields; Mavic Fly-app SRTs provide GPS/altitude/focal only,
// enterprise SRTs (M30/M300/M350) add gb_yaw/gb_pitch/gb_roll gimbal attitude.
// The 35mm-equivalent focal_len is per frame, so zoom ramps are captured.
#include "ot/telemetry.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ot {

namespace {
// Finds "<key>" in s (e.g. "latitude"), skips optional spaces and one ':',
// then parses a double. Returns false if the key or number is absent.
bool find_num(const std::string& s, const char* key, double& out) {
    const size_t p = s.find(key);
    if (p == std::string::npos) return false;
    size_t q = p + std::strlen(key);
    while (q < s.size() && s[q] == ' ') ++q;
    if (q < s.size() && s[q] == ':') ++q;
    while (q < s.size() && s[q] == ' ') ++q;
    char* end = nullptr;
    const double v = std::strtod(s.c_str() + q, &end);
    if (end == s.c_str() + q) return false;
    out = v;
    return true;
}

// "2026-04-25 13:17:52.266" somewhere in the block -> copied to out.
bool find_timestamp(const std::string& s, std::string& out) {
    for (size_t p = 0; p + 19 <= s.size(); ++p) {
        if (!std::isdigit(static_cast<unsigned char>(s[p]))) continue;
        if (s[p + 4] != '-' || s[p + 7] != '-' || s[p + 10] != ' ' ||
            s[p + 13] != ':' || s[p + 16] != ':')
            continue;
        size_t q = p + 19;                       // optional .mmm fraction
        if (q < s.size() && s[q] == '.') {
            ++q;
            while (q < s.size() && std::isdigit(static_cast<unsigned char>(s[q]))) ++q;
        }
        out.assign(s, p, q - p);
        return true;
    }
    return false;
}

// One subtitle block (the lines between blank lines) -> telemetry record.
bool parse_block(const std::string& text, TeleFrame& out) {
    double frame = 0;
    if (!find_num(text, "FrameCnt", frame)) return false;
    out.frame = static_cast<int>(frame);
    find_timestamp(text, out.time);

    double lat = 0, lon = 0;
    out.has_gps = find_num(text, "latitude", lat) && find_num(text, "longitude", lon);
    if (out.has_gps) { out.lat = lat; out.lon = lon; }
    find_num(text, "rel_alt", out.rel_alt);
    find_num(text, "abs_alt", out.abs_alt);
    out.has_focal = find_num(text, "focal_len", out.focal_len);
    out.has_attitude = find_num(text, "gb_pitch", out.gb_pitch);
    find_num(text, "gb_yaw", out.gb_yaw);
    find_num(text, "gb_roll", out.gb_roll);
    return true;
}
}  // namespace

std::vector<TeleFrame> parse_srt(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("parse_srt: cannot open '" + path + "'");

    std::vector<TeleFrame> frames;
    std::string line, block;
    auto flush = [&] {
        TeleFrame f;
        if (!block.empty() && parse_block(block, f)) frames.push_back(std::move(f));
        block.clear();
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) { flush(); continue; }
        block += line;
        block += ' ';
    }
    flush();
    if (frames.empty())
        throw std::runtime_error("parse_srt: no telemetry blocks in '" + path + "'");
    return frames;
}

std::string find_srt_sidecar(const std::string& video_path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(video_path, ec)) return "";
    std::filesystem::path p(video_path);
    for (const char* ext : {".SRT", ".srt"}) {
        std::filesystem::path cand = p;
        cand.replace_extension(ext);
        if (std::filesystem::exists(cand, ec)) return cand.string();
    }
    return "";
}

}  // namespace ot
