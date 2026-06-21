#pragma once

#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector.hpp"
#include "ot/inference_backend.hpp"

#include <memory>

namespace ot {

// Detector for the RF-DETR output family: a transformer detector that emits a
// fixed set of object queries (no anchors, no NMS). The exported ONNX takes
// ImageNet-normalized RGB NCHW input (stretch-resized to a square — NOT
// letterboxed) and returns two tensors:
//   boxes  [1, Q, 4]  cxcywh, normalized to [0,1]
//   labels [1, Q, C]  raw class logits (apply sigmoid)
// Decode: for each query keep its best people/vehicle class whose sigmoid score
// passes `conf`, convert cxcywh->xyxy, and scale to the frame. Class filtering
// uses the configured ClassMap (RF-DETR emits COCO ids).
class RfDetrDetector : public Detector {
public:
    explicit RfDetrDetector(const DetectorCfg& cfg);
    std::vector<Detection> detect(const cv::Mat& frame) override;
    // One batched inference over all images (SAHI tiles + full frame). Requires a
    // dynamic-batch ONNX (export with --dynamic-batch, the default). Falls back to
    // per-image inference if the batched run fails (e.g. fixed-batch model / OOM).
    std::vector<std::vector<Detection>> detect_batch(const std::vector<cv::Mat>& imgs) override;

private:
    DetectorCfg                        cfg_;
    ClassMap                           class_map_;
    std::vector<int>                   kept_ids_;     // people/vehicle native ids (hot-loop)
    float                              logit_thresh_; // inverse-sigmoid(conf): skip sigmoid below this
    std::unique_ptr<InferenceBackend>  backend_;

    // Stretch-resize one frame to an input_size square and ImageNet-normalize it
    // directly into the CHW slot at `dst` (3*S*S floats). No letterbox — boxes
    // come back normalized to this stretched space, so undoing the resize is just
    // a multiply by the frame's own width/height.
    void preprocess_into(const cv::Mat& frame, float* dst) const;
    std::vector<Detection> decode_one(const float* boxes, const float* logits,
                                      int Q, int C, int frame_w, int frame_h) const;
};

}  // namespace ot
