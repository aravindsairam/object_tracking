// TargetSink implementations + factory. JsonlSink appends one JSON object per
// frame (JSON Lines) so a downstream control loop can stream/parse the locked
// target; NullSink discards everything. make_sink picks one from the
// output.sink config string ("jsonl" | "none").
#include "ot/target_sink.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace ot {

namespace {
const char* state_name(LockState s) {
    switch (s) {
        case LockState::Acquiring: return "acquiring";
        case LockState::Locked:    return "locked";
        case LockState::Coasting:  return "coasting";
        case LockState::Lost:      return "lost";
    }
    return "unknown";
}

// Writes one JSON object per line (JSON Lines) — easy to stream/parse downstream.
class JsonlSink : public TargetSink {
public:
    explicit JsonlSink(const std::string& path) : out_(path, std::ios::trunc) {
        if (!out_) throw std::runtime_error("JsonlSink: cannot open '" + path + "'");
    }
    void write(const TargetState& s) override {
        nlohmann::json j = {
            {"frame", s.frame_index},
            {"t", s.timestamp_s},
            {"state", state_name(s.state)},
            {"box", {s.box.x, s.box.y, s.box.w, s.box.h}},
            {"center_px", {s.center_px.x, s.center_px.y}},
            {"center_norm", {s.center_norm.x, s.center_norm.y}},
            {"vel_px_s", {s.velocity_px_s.x, s.velocity_px_s.y}},
            {"conf", s.confidence},
            {"class_id", s.class_id},     // voted/stable class of the locked target
            {"lock_id", s.lock_id},       // stable id for this lock (kept across re-acquire)
            {"track_id", s.track_id},     // raw MOT id (churns across re-acquires)
        };
        out_ << j.dump() << '\n';
        out_.flush();
    }

private:
    std::ofstream out_;
};

class NullSink : public TargetSink {
public:
    void write(const TargetState&) override {}
};
}  // namespace

std::unique_ptr<TargetSink> make_sink(const std::string& kind, const std::string& path) {
    if (kind == "jsonl") return std::make_unique<JsonlSink>(path);
    if (kind == "none" || kind.empty()) return std::make_unique<NullSink>();
    throw std::runtime_error("make_sink: unknown sink '" + kind + "' (expected jsonl | none)");
}

}  // namespace ot
