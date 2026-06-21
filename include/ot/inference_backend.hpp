#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ot {

// A TensorRT dynamic-shape optimization profile. With a dynamic batch dim, the
// TRT EP needs explicit min/opt/max shapes or it builds a fresh engine per batch
// size it encounters (e.g. a ~minutes-long stall the first time the locked-ROI
// path runs batch=1 after the SAHI path built a batch=8 engine). One profile
// spanning [min,max] lets a single engine serve every batch in range; `opt` is
// the batch the kernels are tuned for. Spatial dims are fixed (c,h,w).
struct TrtProfile {
    std::string input_name;            // model input tensor name (RF-DETR: "input")
    int c = 0, h = 0, w = 0;           // fixed channel + spatial dims
    int min_batch = 0, opt_batch = 0, max_batch = 0;
    bool enabled() const {
        return !input_name.empty() && c > 0 && h > 0 && w > 0 &&
               min_batch >= 1 && opt_batch >= min_batch && max_batch >= opt_batch;
    }
};

// Runs a preprocessed NCHW blob through a network and returns the raw output
// tensor as a cv::Mat (with the model's output dims). This is the "backend
// axis": the same detector decode runs on OpenCV-dnn, ONNX Runtime, or
// TensorRT — only this class changes.
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // `blob_nchw` is a 4D CV_32F Mat [1,3,H,W] (e.g. from cv::dnn::blobFromImage).
    // Returns the first/only output tensor, dims preserved.
    virtual cv::Mat run(const cv::Mat& blob_nchw) = 0;

    // Returns ALL output tensors in model order (single-output models give one).
    // Multi-head models (e.g. RF-DETR: boxes + labels) override/use this. Default
    // wraps run() so single-output backends need not implement it.
    virtual std::vector<cv::Mat> run_multi(const cv::Mat& blob_nchw) {
        return {run(blob_nchw)};
    }

    virtual std::string name() const = 0;
};

// Builds a backend for the given `kind`:
//   "onnxruntime" | "ort"   -> ONNX Runtime (CUDA EP if device=="cuda", else CPU)
//   "tensorrt" | "trt"      -> ONNX Runtime TensorRT EP (with CUDA/CPU fallback)
//   "ocv_dnn"               -> unsupported (throws)
// `precision` (fp32|fp16|int8) applies to the TensorRT EP only. `int8_calib` is
// a path to an INT8 calibration table, required when precision=="int8". `profile`
// (TensorRT EP only) pins the dynamic-shape optimization profile; pass a default
// (disabled) TrtProfile for fixed-shape or non-TRT models. Throws
// std::runtime_error on failure/unknown kind.
std::unique_ptr<InferenceBackend> make_backend(const std::string& kind,
                                               const std::string& model_path,
                                               const std::string& device = "cpu",
                                               const std::string& precision = "fp32",
                                               const std::string& int8_calib = "",
                                               const TrtProfile& profile = {});

}  // namespace ot
