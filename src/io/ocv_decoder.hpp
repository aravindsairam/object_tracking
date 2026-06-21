#pragma once

#include "ot/frame_decoder.hpp"

#include <opencv2/videoio.hpp>

#include <string>

namespace ot {

// Software (or OpenCV-managed) decode via cv::VideoCapture: any file / RTSP URL /
// camera index / GStreamer pipeline OpenCV understands. The portable fallback
// path; also handles camera indices and explicit GStreamer pipelines that the
// NVDEC path does not.
class OcvDecoder : public FrameDecoder {
public:
    OcvDecoder(const std::string& source, int target_height);

    bool read(cv::Mat& out) override;
    bool seek(int64_t index) override;

    double  fps() const override { return fps_; }
    int     width() const override { return width_; }
    int     height() const override { return height_; }
    int     native_width() const override { return native_width_; }
    int     native_height() const override { return native_height_; }
    int64_t frame_index() const override { return frame_index_; }

private:
    cv::VideoCapture cap_;
    double  fps_ = 0.0;
    int     width_ = 0;
    int     height_ = 0;
    int     native_width_ = 0;
    int     native_height_ = 0;
    bool    resize_ = false;
    int64_t frame_index_ = -1;
};

}  // namespace ot
