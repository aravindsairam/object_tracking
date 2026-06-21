#pragma once

#include "ot/frame_decoder.hpp"

#include <opencv2/core.hpp>
#include <cstdint>
#include <memory>
#include <string>

namespace ot {

// Decodes frames from any video file / RTSP URL / camera index, downscaled to a
// working resolution. A thin selector over a FrameDecoder backend: NVDEC
// hardware decode (NvdecDecoder) when possible, else OpenCV software decode
// (OcvDecoder). The rest of the system only ever sees plain BGR cv::Mat frames.
class VideoSource {
public:
    // Opens `source` (file path, RTSP URL, or numeric camera index as string).
    // If `target_height > 0`, every frame is downscaled (aspect preserved) to
    // that height — the system's working resolution; all downstream coordinates
    // live in this scaled space. 0 = keep native. Never upscales.
    // `decoder` selects the backend: "auto" (NVDEC with software fallback),
    // "nvdec" (require NVDEC), or "software". The OT_DECODER env var overrides it.
    // Throws std::runtime_error if the source cannot be opened.
    explicit VideoSource(const std::string& source, int target_height = 0,
                         const std::string& decoder = "auto");

    bool read(cv::Mat& out)   { return dec_->read(out); }
    bool seek(int64_t index)  { return dec_->seek(index); }

    double  fps() const           { return dec_->fps(); }
    int     width() const         { return dec_->width(); }
    int     height() const        { return dec_->height(); }
    int     native_width() const  { return dec_->native_width(); }
    int     native_height() const { return dec_->native_height(); }
    int64_t frame_index() const   { return dec_->frame_index(); }

private:
    std::unique_ptr<FrameDecoder> dec_;
};

}  // namespace ot
