// Detector for the YOLO output family (YOLO / WALDO / UniDrone). Letterboxes
// the input to a square, runs the backend (batched so SAHI tiles share one
// inference call), then decodes either a raw anchor-free head (+NMS here) or an
// NMS-embedded head into Detections. The class_map filters to person/vehicle.
#include "yolo_detector.hpp"

#include <opencv2/dnn.hpp>      // blobFromImage (preprocessing only)
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

namespace ot {

YoloDetector::YoloDetector(const DetectorCfg& cfg)
    : cfg_(cfg),
      class_map_(ClassMap::preset(cfg.class_map)),
      backend_(make_backend(cfg.backend, cfg.model_path, cfg.device,
                            cfg.precision, cfg.int8_calib)) {}

float YoloDetector::letterbox(const cv::Mat& frame, cv::Mat& out) const {
    const int s = cfg_.input_size;
    const float scale = std::min(static_cast<float>(s) / frame.cols,
                                 static_cast<float>(s) / frame.rows);
    const int nw = static_cast<int>(std::round(frame.cols * scale));
    const int nh = static_cast<int>(std::round(frame.rows * scale));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    out = cv::Mat(s, s, frame.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(0, 0, nw, nh)));  // top-left aligned
    return scale;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Mat> one{frame};
    return std::move(detect_batch(one).front());
}

std::vector<std::vector<Detection>> YoloDetector::detect_batch(const std::vector<cv::Mat>& imgs) {
    const int N = static_cast<int>(imgs.size());
    std::vector<std::vector<Detection>> results(N);
    if (N == 0) return results;

    // Build one batched blob [N,3,S,S]. Preprocessing (letterbox + HWC->CHW float
    // convert) is the dominant cost for many tiles, so parallelize it per-image.
    const int S = cfg_.input_size;
    const int sizes[4] = {N, 3, S, S};
    cv::Mat blob(4, sizes, CV_32F);
    std::vector<float> scales(N);

    cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& rng) {
        cv::Mat ch;  // reused scratch per thread
        for (int n = rng.start; n < rng.end; ++n) {
            cv::Mat canvas;                       // S×S, 8UC3 BGR
            scales[n] = letterbox(imgs[n], canvas);
            // Write directly into blob planes as float RGB /255 (no temp blob, no memcpy).
            for (int c = 0; c < 3; ++c) {
                cv::Mat dst(S, S, CV_32F, blob.ptr<float>() + (static_cast<size_t>(n) * 3 + c) * S * S);
                cv::extractChannel(canvas, ch, 2 - c);   // BGR -> RGB plane order
                ch.convertTo(dst, CV_32F, 1.0 / 255.0);
            }
        }
    });

    cv::Mat out;
    try {
        out = backend_->run(blob);
    } catch (const std::exception&) {
        if (N == 1) throw;  // genuine failure
        for (int n = 0; n < N; ++n) results[n] = detect(imgs[n]);  // fixed-batch fallback
        return results;
    }

    // out is [N, d1, d2]; slice each image's block and decode independently.
    int d1, d2;
    if (out.dims == 3) { d1 = out.size[1]; d2 = out.size[2]; }
    else if (out.dims == 2) { d1 = 1; d2 = out.size[1]; }  // degenerate single
    else throw std::runtime_error("YoloDetector: unexpected output rank " + std::to_string(out.dims));

    const bool embedded = (d2 == 6 && d1 <= 4096);
    const size_t step = static_cast<size_t>(d1) * static_cast<size_t>(d2);
    float* base = out.ptr<float>();

    for (int n = 0; n < N; ++n) {
        cv::Mat block(d1, d2, CV_32F, base + n * step);
        results[n] = embedded
            ? decode_nms_embedded(block, scales[n], imgs[n].cols, imgs[n].rows)
            : decode_raw(block, scales[n], imgs[n].cols, imgs[n].rows);
    }
    return results;
}

std::vector<Detection> YoloDetector::decode_raw(const cv::Mat& out, float scale,
                                                int frame_w, int frame_h) const {
    int a, b;
    if (out.dims == 3) { a = out.size[1]; b = out.size[2]; }
    else { a = out.size[0]; b = out.size[1]; }

    // View the data as a 2D matrix, then orient to (num_anchors, 4+nc).
    cv::Mat flat(a, b, CV_32F, const_cast<float*>(out.ptr<float>()));
    cv::Mat pred = (a < b) ? flat.t() : flat;   // channel-major [C,N] -> transpose to [N,C]
    const int C = pred.cols;                     // 4 + nc
    const int nc = C - 4;
    if (nc <= 0) throw std::runtime_error("YoloDetector: bad raw output channels");

    std::vector<cv::Rect> boxes;
    std::vector<float>     scores;
    std::vector<int>       classes;
    boxes.reserve(256); scores.reserve(256); classes.reserve(256);

    for (int n = 0; n < pred.rows; ++n) {
        const float* row = pred.ptr<float>(n);
        // argmax over class scores
        int best = 0; float best_s = row[4];
        for (int c = 1; c < nc; ++c) {
            if (row[4 + c] > best_s) { best_s = row[4 + c]; best = c; }
        }
        if (best_s < cfg_.conf || !class_map_.keep(best)) continue;

        const float cx = row[0], cy = row[1], w = row[2], h = row[3];
        const float x1 = (cx - w * 0.5f) / scale;
        const float y1 = (cy - h * 0.5f) / scale;
        boxes.emplace_back(cvRound(x1), cvRound(y1), cvRound(w / scale), cvRound(h / scale));
        scores.push_back(best_s);
        classes.push_back(best);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, cfg_.conf, cfg_.nms_iou, keep);

    std::vector<Detection> dets;
    dets.reserve(keep.size());
    for (int i : keep) {
        Detection d;
        d.box = {static_cast<float>(boxes[i].x), static_cast<float>(boxes[i].y),
                 static_cast<float>(boxes[i].width), static_cast<float>(boxes[i].height)};
        d.box.clamp(frame_w, frame_h);
        d.class_id = classes[i];
        d.score = scores[i];
        dets.push_back(d);
    }
    return dets;
}

std::vector<Detection> YoloDetector::decode_nms_embedded(const cv::Mat& out, float scale,
                                                         int frame_w, int frame_h) const {
    const int M = (out.dims == 3) ? out.size[1] : out.size[0];
    cv::Mat rows(M, 6, CV_32F, const_cast<float*>(out.ptr<float>()));

    std::vector<Detection> dets;
    for (int i = 0; i < M; ++i) {
        const float* r = rows.ptr<float>(i);
        const float score = r[4];
        const int   cls   = static_cast<int>(r[5]);
        if (score < cfg_.conf || !class_map_.keep(cls)) continue;
        Detection d;
        d.box = {r[0] / scale, r[1] / scale, (r[2] - r[0]) / scale, (r[3] - r[1]) / scale};
        d.box.clamp(frame_w, frame_h);
        d.class_id = cls;
        d.score = score;
        dets.push_back(d);
    }
    return dets;
}

}  // namespace ot
