#pragma once

#include "ot/inference_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace ot {

// ONNX Runtime inference backend. Execution-provider priority is set at
// construction: TensorRT EP (with fp16/int8) -> CUDA EP -> CPU. Each provider is
// best-effort: if it can't initialize, we fall through to the next so a missing
// GPU/TensorRT never crashes the app.
class OrtBackend : public InferenceBackend {
public:
    // `use_cuda` appends the CUDA EP; `use_tensorrt` appends the TensorRT EP
    // ahead of CUDA. `precision` (fp32|fp16|int8) and `int8_calib` apply to the
    // TensorRT EP only.
    OrtBackend(const std::string& model_path, bool use_cuda, bool use_tensorrt,
               const std::string& precision, const std::string& int8_calib,
               const TrtProfile& profile = {});
    cv::Mat run(const cv::Mat& blob_nchw) override;
    std::vector<cv::Mat> run_multi(const cv::Mat& blob_nchw) override;
    std::string name() const override { return name_; }

private:
    std::string              int8_calib_;  // kept alive for the TRT EP option
    Ort::Env                 env_;
    Ort::SessionOptions      opts_;
    Ort::Session             session_{nullptr};
    Ort::AllocatorWithDefaultOptions alloc_;
    std::string              name_ = "onnxruntime(cpu)";
    std::vector<std::string> input_name_store_;
    std::vector<std::string> output_name_store_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Inference profiling (enabled when the OT_PROFILE env var is set). Times the
    // pure session_.Run (GPU compute + H2D/D2H copies) so it can be compared
    // against the wider detect() stage to see how much is host-side work.
    bool   profile_      = false;
    double infer_ms_sum_ = 0.0;
    long   infer_calls_  = 0;
};

}  // namespace ot
