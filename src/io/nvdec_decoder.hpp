#pragma once

#include "ot/frame_decoder.hpp"

#include <string>

// Opaque libav types — keep the FFmpeg headers out of this header.
struct AVFormatContext;
struct AVCodecContext;
struct AVPacket;
struct AVFrame;
struct SwsContext;

namespace ot {

// Hardware video decode via FFmpeg libavcodec + NVIDIA cuvid (NVDEC). Decodes
// H.264/HEVC/AV1/VP9 files & RTSP streams on the GPU, GPU-resizes to the working
// resolution during decode (cuvid `resize` option), and returns BGR frames.
// Throws std::runtime_error from the constructor if NVDEC cannot be used for
// this input (no GPU/cuvid, unsupported codec, unreadable source) — the caller
// (VideoSource) then falls back to software decode.
class NvdecDecoder : public FrameDecoder {
public:
    NvdecDecoder(const std::string& source, int target_height);
    ~NvdecDecoder() override;

    NvdecDecoder(const NvdecDecoder&) = delete;
    NvdecDecoder& operator=(const NvdecDecoder&) = delete;

    bool read(cv::Mat& out) override;
    bool seek(int64_t index) override;

    double  fps() const override { return fps_; }
    int     width() const override { return width_; }
    int     height() const override { return height_; }
    int     native_width() const override { return native_width_; }
    int     native_height() const override { return native_height_; }
    int64_t frame_index() const override { return frame_index_; }

private:
    AVFormatContext* fmt_   = nullptr;
    AVCodecContext*  dec_   = nullptr;
    AVPacket*        pkt_   = nullptr;
    AVFrame*         frame_ = nullptr;
    SwsContext*      sws_   = nullptr;

    int      video_stream_ = -1;
    double   fps_ = 30.0;
    int      width_ = 0, height_ = 0, native_width_ = 0, native_height_ = 0;
    int64_t  frame_index_ = -1;
    bool     draining_ = false;   // input EOF reached; decoder being flushed
};

}  // namespace ot
