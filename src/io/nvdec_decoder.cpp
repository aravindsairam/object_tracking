// Hardware video decode via FFmpeg's NVDEC (cuvid) decoders — H.264/HEVC/AV1/
// VP9/MPEG-2 — with optional GPU-side downscale to the working height. Yields
// BGR cv::Mat frames. Construction throws when the codec has no NVDEC decoder
// or no capable GPU is present; VideoSource catches that and falls back to the
// software (OpenCV) decoder.
#include "nvdec_decoder.hpp"

#include <opencv2/core.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ot {

namespace {
std::runtime_error fail(const std::string& msg) {
    return std::runtime_error("NvdecDecoder: " + msg);
}
}  // namespace

NvdecDecoder::NvdecDecoder(const std::string& source, int target_height) {
    av_log_set_level(AV_LOG_ERROR);  // quiet libav's chatter; keep real errors

    if (avformat_open_input(&fmt_, source.c_str(), nullptr, nullptr) < 0)
        throw fail("cannot open '" + source + "'");
    if (avformat_find_stream_info(fmt_, nullptr) < 0)
        throw fail("no stream info for '" + source + "'");

    video_stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_ < 0) throw fail("no video stream");

    AVStream* st = fmt_->streams[video_stream_];
    const AVCodecParameters* par = st->codecpar;
    native_width_  = par->width;
    native_height_ = par->height;

    // Map the stream codec to its NVDEC (cuvid) decoder.
    const char* dec_name = nullptr;
    switch (par->codec_id) {
        case AV_CODEC_ID_HEVC:  dec_name = "hevc_cuvid";  break;
        case AV_CODEC_ID_H264:  dec_name = "h264_cuvid";  break;
        case AV_CODEC_ID_AV1:   dec_name = "av1_cuvid";   break;
        case AV_CODEC_ID_VP9:   dec_name = "vp9_cuvid";   break;
        case AV_CODEC_ID_MPEG2VIDEO: dec_name = "mpeg2_cuvid"; break;
        default: throw fail("codec has no NVDEC decoder");
    }
    const AVCodec* codec = avcodec_find_decoder_by_name(dec_name);
    if (!codec) throw fail(std::string(dec_name) + " unavailable (FFmpeg built without cuvid?)");

    dec_ = avcodec_alloc_context3(codec);
    if (!dec_) throw fail("avcodec_alloc_context3 failed");
    if (avcodec_parameters_to_context(dec_, par) < 0) throw fail("parameters_to_context failed");
    dec_->pkt_timebase = st->time_base;

    // Working resolution: downscale to target_height (never upscale), even dims.
    // cuvid's `resize` option does this scaling on the GPU during decode.
    if (target_height > 0 && native_height_ > 0 && target_height < native_height_) {
        height_ = target_height & ~1;
        width_  = static_cast<int>(std::lround(
                      native_width_ * (static_cast<double>(height_) / native_height_))) & ~1;
        char res[32];
        std::snprintf(res, sizeof(res), "%dx%d", width_, height_);
        av_opt_set(dec_->priv_data, "resize", res, 0);
    } else {
        width_  = native_width_;
        height_ = native_height_;
    }

    if (avcodec_open2(dec_, codec, nullptr) < 0)
        throw fail("cannot open " + std::string(dec_name) + " (no NVDEC-capable GPU?)");

    pkt_   = av_packet_alloc();
    frame_ = av_frame_alloc();
    if (!pkt_ || !frame_) throw fail("packet/frame alloc failed");

    const AVRational fr = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    fps_ = (fr.num && fr.den) ? static_cast<double>(fr.num) / fr.den : 30.0;
    if (fps_ <= 1.0 || fps_ > 1000.0) fps_ = 30.0;
}

NvdecDecoder::~NvdecDecoder() {
    if (sws_)   sws_freeContext(sws_);
    if (frame_) av_frame_free(&frame_);
    if (pkt_)   av_packet_free(&pkt_);
    if (dec_)   avcodec_free_context(&dec_);
    if (fmt_)   avformat_close_input(&fmt_);
}

bool NvdecDecoder::read(cv::Mat& out) {
    for (;;) {
        const int ret = avcodec_receive_frame(dec_, frame_);
        if (ret == 0) {
            // cuvid returns NV12 (system memory) at the GPU-resized resolution.
            const int W = frame_->width, H = frame_->height;
            sws_ = sws_getCachedContext(
                sws_, W, H, static_cast<AVPixelFormat>(frame_->format),
                W, H, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws_) { av_frame_unref(frame_); return false; }
            out.create(H, W, CV_8UC3);
            uint8_t* dst[4]     = { out.data, nullptr, nullptr, nullptr };
            int      dstride[4] = { static_cast<int>(out.step[0]), 0, 0, 0 };
            sws_scale(sws_, frame_->data, frame_->linesize, 0, H, dst, dstride);
            ++frame_index_;
            av_frame_unref(frame_);
            return true;
        }
        if (ret == AVERROR_EOF) return false;            // fully drained
        if (ret != AVERROR(EAGAIN)) return false;         // decode error -> treat as EOS
        if (draining_) return false;                      // flushed, nothing left

        // Feed packets until one video packet is sent (or input EOF -> flush).
        for (bool sent = false; !sent;) {
            if (av_read_frame(fmt_, pkt_) < 0) {
                avcodec_send_packet(dec_, nullptr);       // enter drain mode
                draining_ = true;
                sent = true;
            } else {
                if (pkt_->stream_index == video_stream_) {
                    avcodec_send_packet(dec_, pkt_);
                    sent = true;
                }
                av_packet_unref(pkt_);
            }
        }
    }
}

bool NvdecDecoder::seek(int64_t index) {
    if (index < 0) index = 0;
    AVStream* st = fmt_->streams[video_stream_];
    const int64_t ts = static_cast<int64_t>(
        std::llround(index / fps_ / av_q2d(st->time_base)));
    if (av_seek_frame(fmt_, video_stream_, ts, AVSEEK_FLAG_BACKWARD) < 0) return false;
    avcodec_flush_buffers(dec_);
    draining_ = false;
    frame_index_ = index - 1;  // next read() yields ~`index` (keyframe-aligned)
    return true;
}

}  // namespace ot
