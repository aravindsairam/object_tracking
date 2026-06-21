#pragma once

#include "ot/class_map.hpp"
#include "ot/config.hpp"
#include "ot/detector.hpp"
#include "ot/inference_backend.hpp"

#include <memory>

namespace ot {

// Detector for the YOLO output family. Loads an ONNX via OpenCV's dnn module
// and auto-handles both common export layouts:
//   * raw head        [1, 4+nc, N]      (YOLOv8/WALDO/UniDrone) -> decode + NMS
//   * NMS-embedded    [1, M, 6]         (YOLO26 end2end)         -> already boxes
// Coordinates are produced in the input frame's pixel space. Class filtering
// uses the configured ClassMap so only people/vehicles are returned.
class YoloDetector : public Detector {
public:
    explicit YoloDetector(const DetectorCfg& cfg);
    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::vector<std::vector<Detection>> detect_batch(const std::vector<cv::Mat>& imgs) override;

private:
    DetectorCfg                        cfg_;
    ClassMap                           class_map_;
    std::unique_ptr<InferenceBackend>  backend_;

    // Letterbox the frame into a square `input_size` canvas (top-left aligned,
    // 114 padding). Returns the scale factor applied (for undoing later).
    float letterbox(const cv::Mat& frame, cv::Mat& out) const;

    std::vector<Detection> decode_raw(const cv::Mat& out, float scale,
                                      int frame_w, int frame_h) const;
    std::vector<Detection> decode_nms_embedded(const cv::Mat& out, float scale,
                                               int frame_w, int frame_h) const;
};

}  // namespace ot
