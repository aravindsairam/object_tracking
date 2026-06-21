#include "ot/video_source.hpp"

#include "nvdec_decoder.hpp"
#include "ocv_decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

namespace ot {

VideoSource::VideoSource(const std::string& source, int target_height,
                         const std::string& decoder) {
    std::string policy = decoder;
    if (const char* env = std::getenv("OT_DECODER")) policy = env;  // quick override

    // Try NVDEC first unless software is explicitly requested. NvdecDecoder
    // throws for inputs it can't handle (camera index, GStreamer pipeline,
    // unsupported codec, no NVDEC GPU) — in "auto" we fall back to software,
    // in "nvdec" we surface the error.
    if (policy != "software") {
        try {
            dec_ = std::make_unique<NvdecDecoder>(source, target_height);
            std::fprintf(stderr, "[video] decoder = nvdec (GPU-resized %dx%d)\n",
                         dec_->width(), dec_->height());
            return;
        } catch (const std::exception& e) {
            if (policy == "nvdec") {
                throw std::runtime_error(
                    std::string("VideoSource: NVDEC required but unavailable: ") + e.what());
            }
            std::fprintf(stderr, "[video] nvdec unavailable (%s) — falling back to software\n",
                         e.what());
        }
    }

    dec_ = std::make_unique<OcvDecoder>(source, target_height);
    std::fprintf(stderr, "[video] decoder = software\n");
}

}  // namespace ot
