#include "ocv_decoder.hpp"

#include <opencv2/imgproc.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ot {

namespace {
// A source that is all digits is treated as a camera index.
bool is_camera_index(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}
}  // namespace

OcvDecoder::OcvDecoder(const std::string& source, int target_height) {
    const bool gst = source.find("appsink") != std::string::npos;
    bool ok = false;
    if (gst) {
        ok = cap_.open(source, cv::CAP_GSTREAMER);
    } else if (is_camera_index(source)) {
        ok = cap_.open(std::stoi(source));
    } else {
        // Opt-in hardware-accelerated decode (NVDEC via FFmpeg's *_cuvid decoders):
        // set OT_HWACCEL=1. Only works if OpenCV's FFmpeg is built with cuvid (the
        // stock apt build is NOT — that's why the dedicated NvdecDecoder exists).
        // Off by default to avoid a noisy fallback. Requires OpenCV >= 4.5.
        if (std::getenv("OT_HWACCEL")) {
            ok = cap_.open(source, cv::CAP_FFMPEG,
                           {cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
        }
        if (!ok || !cap_.isOpened()) ok = cap_.open(source);  // software (default)
    }
    if (!ok || !cap_.isOpened()) {
        throw std::runtime_error("OcvDecoder: failed to open source '" + source + "'");
    }

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 1.0 || fps_ > 1000.0) fps_ = 30.0;  // sane fallback for streams

    native_width_  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    native_height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Compute working resolution: downscale to target_height, never upscale.
    if (target_height > 0 && native_height_ > 0 && target_height < native_height_) {
        resize_ = true;
        height_ = target_height;
        width_  = static_cast<int>(std::lround(
            native_width_ * (static_cast<double>(target_height) / native_height_)));
    } else {
        width_  = native_width_;
        height_ = native_height_;
    }
}

bool OcvDecoder::read(cv::Mat& out) {
    cv::Mat raw;
    if (!cap_.read(raw) || raw.empty()) return false;
    if (resize_) {
        cv::resize(raw, out, cv::Size(width_, height_), 0, 0, cv::INTER_AREA);
    } else {
        out = raw;
    }
    ++frame_index_;
    return true;
}

bool OcvDecoder::seek(int64_t index) {
    if (index < 0) index = 0;
    if (!cap_.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(index))) return false;
    frame_index_ = index - 1;  // next read() yields `index`
    return true;
}

}  // namespace ot
