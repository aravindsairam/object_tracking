#pragma once

#include <opencv2/core.hpp>

#include <cstdint>

namespace ot {

// Decodes frames from a video file / RTSP URL / camera into BGR cv::Mat at a
// working resolution. The "decode axis": the same VideoSource/pipeline runs on
// top of OpenCV/FFmpeg software decode (OcvDecoder) or NVIDIA NVDEC hardware
// decode (NvdecDecoder) — only this implementation changes. All coordinates the
// rest of the system sees live in the working-resolution space.
class FrameDecoder {
public:
    virtual ~FrameDecoder() = default;

    // Reads the next frame into `out` (BGR, working resolution). False at EOS.
    virtual bool read(cv::Mat& out) = 0;

    // Seeks so the next read() returns frame `index` (clamped at 0). Best-effort
    // (keyframe-aligned) for formats without exact seek. False if unsupported.
    virtual bool seek(int64_t index) = 0;

    virtual double  fps() const = 0;
    virtual int     width() const = 0;          // working (output) width
    virtual int     height() const = 0;         // working (output) height
    virtual int     native_width() const = 0;
    virtual int     native_height() const = 0;
    virtual int64_t frame_index() const = 0;    // index of last frame read
};

}  // namespace ot
