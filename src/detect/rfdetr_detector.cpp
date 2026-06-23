#include "rfdetr_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ot {

namespace {
// RF-DETR preprocessing: ImageNet mean/std on RGB in [0,1].
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

// TRT optimization-profile batch range (used both to build the engine and to
// chunk oversized batches in detect_batch). The engine accepts batches in
// [kMinBatch, kMaxBatch], tuned for kOptBatch; anything larger is split.
constexpr int kMinBatch = 1;
constexpr int kOptBatch = 8;
constexpr int kMaxBatch = 16;

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Inverse sigmoid: the raw-logit value whose sigmoid equals `p`. Lets the decode
// loop threshold on the logit directly (sigmoid is monotonic) and only call the
// expensive sigmoid on queries that actually pass `conf`.
inline float inverse_sigmoid(float p) {
    p = std::min(std::max(p, 1e-6f), 1.0f - 1e-6f);
    return std::log(p / (1.0f - p));
}
}  // namespace

RfDetrDetector::RfDetrDetector(const DetectorCfg& cfg)
    : cfg_(cfg),
      class_map_(ClassMap::preset(cfg.class_map)),
      kept_ids_(class_map_.kept_ids()),
      logit_thresh_(inverse_sigmoid(cfg.conf)),
      // TRT optimization profile: one engine spanning batch kMinBatch (locked-ROI
      // path) .. kMaxBatch (SAHI tiles + headroom), tuned for kOptBatch. Tile sets
      // larger than kMaxBatch (small-tile/high-res SAHI) are chunked in
      // detect_batch. Input is fixed 3xSxS; tensor name "input" per the ONNX contract.
      backend_(make_backend(cfg.backend, cfg.model_path, cfg.device,
                            cfg.precision, cfg.int8_calib,
                            TrtProfile{"input", 3, cfg.input_size, cfg.input_size,
                                       kMinBatch, kOptBatch, kMaxBatch})) {}

void RfDetrDetector::preprocess_into(const cv::Mat& frame, float* dst) const {
    const int S = cfg_.input_size;
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(S, S), 0, 0, cv::INTER_LINEAR);  // stretch, no letterbox
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat chan;
    for (int c = 0; c < 3; ++c) {  // c: 0=R, 1=G, 2=B
        cv::Mat plane(S, S, CV_32F, dst + static_cast<size_t>(c) * S * S);
        cv::extractChannel(rgb, chan, c);
        // Fold /255 + ImageNet (mean,std) into a single scale+shift pass:
        //   (x/255 - mean)/std  ==  x * (1/(255*std))  +  (-mean/std)
        // One convertTo instead of convert + subtract + divide — drops two
        // element-wise passes and the MatExpr temporary per channel. Algebraically
        // identical (verified bit-parity via rfdetr_smoke).
        const double alpha = 1.0 / (255.0 * static_cast<double>(kStd[c]));
        const double beta  = -static_cast<double>(kMean[c]) / static_cast<double>(kStd[c]);
        chan.convertTo(plane, CV_32F, alpha, beta);
    }
}

std::vector<Detection> RfDetrDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Mat> one{frame};
    return std::move(detect_batch(one).front());
}

std::vector<std::vector<Detection>> RfDetrDetector::detect_batch(const std::vector<cv::Mat>& imgs) {
    const int N = static_cast<int>(imgs.size());
    std::vector<std::vector<Detection>> results(N);
    if (N == 0) return results;

    // The TRT engine's optimization profile spans batch [kMinBatch, kMaxBatch]
    // (see ctor); a larger batch fails setInputShape. Small-tile / high-res SAHI
    // configs can emit 40+ tiles in one set, so split anything over kMaxBatch into
    // sub-batches and stitch the results back in order. RF-DETR is compute-bound
    // per tile, so chunking is perf-neutral, and it caps the input-blob memory at
    // kMaxBatch tiles regardless of how finely the frame is sliced.
    if (N > kMaxBatch) {
        for (int start = 0; start < N; start += kMaxBatch) {
            const int end = std::min(start + kMaxBatch, N);
            std::vector<cv::Mat> chunk(imgs.begin() + start, imgs.begin() + end);
            auto chunk_res = detect_batch(chunk);  // each chunk is <= kMaxBatch
            for (int i = start; i < end; ++i)
                results[i] = std::move(chunk_res[i - start]);
        }
        return results;
    }

    // One [N,3,S,S] blob; preprocess each image in parallel (dominant cost for
    // many SAHI tiles). Each tile is stretch-resized to S×S, so variable tile
    // sizes pack into a uniform batch.
    const int S = cfg_.input_size;
    const int sizes[4] = {N, 3, S, S};
    // Reuse the input buffer across calls: create() only reallocates when the
    // shape changes (e.g. ROI batch=1 vs SAHI batch=8), so the high-frequency
    // ROI path stops malloc/free-ing a multi-MB blob every frame.
    blob_.create(4, sizes, CV_32F);
    cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& rng) {
        for (int n = rng.start; n < rng.end; ++n) {
            preprocess_into(imgs[n], blob_.ptr<float>() + static_cast<size_t>(n) * 3 * S * S);
        }
    });

    std::vector<cv::Mat> outs;
    try {
        outs = backend_->run_multi(blob_);
    } catch (const std::exception&) {
        if (N == 1) throw;                                   // genuine failure
        for (int n = 0; n < N; ++n) results[n] = detect(imgs[n]);  // fixed-batch / OOM fallback
        return results;
    }
    if (outs.size() < 2) {
        throw std::runtime_error("RfDetrDetector: expected 2 outputs (boxes, labels), got "
                                 + std::to_string(outs.size()));
    }

    // Identify outputs by last-dim: boxes end in 4, logits in the class count.
    const cv::Mat* boxes = nullptr;
    const cv::Mat* logits = nullptr;
    for (const auto& o : outs) {
        if (o.dims < 3) continue;                            // expect [N,Q,*]
        if (o.size[o.dims - 1] == 4) boxes = &o;
        else                         logits = &o;
    }
    if (!boxes || !logits) {
        throw std::runtime_error("RfDetrDetector: could not identify boxes/logits outputs");
    }
    const int Q = boxes->size[boxes->dims - 2];
    const int C = logits->size[logits->dims - 1];
    if (boxes->size[0] != N || logits->size[0] != N) {
        throw std::runtime_error("RfDetrDetector: output batch != input batch (need a "
                                 "dynamic-batch ONNX; re-export with --dynamic-batch)");
    }

    const float* bbase = boxes->ptr<float>();
    const float* lbase = logits->ptr<float>();
    for (int n = 0; n < N; ++n) {
        results[n] = decode_one(bbase + static_cast<size_t>(n) * Q * 4,
                                lbase + static_cast<size_t>(n) * Q * C,
                                Q, C, imgs[n].cols, imgs[n].rows);
    }
    return results;
}

std::vector<Detection> RfDetrDetector::decode_one(const float* bptr, const float* lptr,
                                                  int Q, int C, int frame_w, int frame_h) const {
    std::vector<Detection> dets;
    dets.reserve(64);
    for (int q = 0; q < Q; ++q) {
        const float* l = lptr + static_cast<size_t>(q) * C;

        // Best people/vehicle class for this query (argmax of logit == argmax of
        // sigmoid, since sigmoid is monotonic). Iterate only the ~6 kept ids
        // directly instead of scanning all C classes with a per-class hash lookup.
        // One detection per query at most.
        int best_c = -1;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int c : kept_ids_) {
            if (c >= C) continue;                       // guard against map/model mismatch
            if (l[c] > best_logit) { best_logit = l[c]; best_c = c; }
        }
        if (best_c < 0) continue;
        if (best_logit < logit_thresh_) continue;       // below conf: skip the sigmoid
        const float s = sigmoid(best_logit);
        const float* b = bptr + static_cast<size_t>(q) * 4;

        // cxcywh normalized -> xyxy pixels. The stretch resize means normalized
        // coords map straight onto the original frame via its own dimensions.
        const float cx = b[0] * frame_w, cy = b[1] * frame_h;
        const float w  = b[2] * frame_w, h  = b[3] * frame_h;
        Detection d;
        d.box = {cx - w * 0.5f, cy - h * 0.5f, w, h};
        d.box.clamp(frame_w, frame_h);
        d.class_id = best_c;
        d.score = s;
        dets.push_back(d);
    }
    return dets;
}

}  // namespace ot
