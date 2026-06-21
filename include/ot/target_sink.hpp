#pragma once

#include "ot/target_state.hpp"

#include <memory>
#include <string>

namespace ot {

// Sink for the per-frame locked-target record. Implementations write it
// somewhere durable (a JSON-lines file now; a network publisher later).
class TargetSink {
public:
    virtual ~TargetSink() = default;
    virtual void write(const TargetState& s) = 0;
};

// Builds a sink for `kind` ("jsonl" | "none"), writing to `path`. "none" returns
// a no-op sink. Throws std::runtime_error if a file sink can't be opened.
std::unique_ptr<TargetSink> make_sink(const std::string& kind, const std::string& path);

}  // namespace ot
