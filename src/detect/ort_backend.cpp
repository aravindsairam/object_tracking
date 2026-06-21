#include "ort_backend.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace ot {

OrtBackend::OrtBackend(const std::string& model_path, bool use_cuda, bool use_tensorrt,
                       const std::string& precision, const std::string& int8_calib,
                       const TrtProfile& profile)
    : int8_calib_(int8_calib),
      env_(ORT_LOGGING_LEVEL_ERROR, "ot") {  // errors only — silence harmless GPU-probe warnings
    profile_ = std::getenv("OT_PROFILE") != nullptr;
    opts_.SetIntraOpNumThreads(0);  // 0 = let ORT pick
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool trt_active = false;
    // EP priority: TensorRT -> CUDA -> CPU. Appending CUDA after TensorRT lets ORT
    // place nodes the TRT EP can't handle on CUDA instead of failing the run.
    if (use_tensorrt) {
        try {
            const OrtApi& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* trt = nullptr;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));  // sane defaults
            std::unique_ptr<OrtTensorRTProviderOptionsV2,
                            void (*)(OrtTensorRTProviderOptionsV2*)>
                trt_guard(trt, api.ReleaseTensorRTProviderOptions);

            // Cache built engines + the kernel timing cache next to the model, by
            // ABSOLUTE path, so the (slow) engine build is reused regardless of the
            // process working directory. A cwd-relative path silently rebuilds the
            // engine on every run launched from a new directory. Must outlive the
            // UpdateTensorRTProviderOptions call below (same scope — fine).
            std::error_code ec;
            std::filesystem::path model_dir = std::filesystem::absolute(model_path, ec).parent_path();
            const std::string engine_cache_path =
                (model_dir.empty() ? std::filesystem::path("trt_engine_cache")
                                   : model_dir / "trt_engine_cache").string();

            std::vector<const char*> keys, vals;
            keys.push_back("device_id");                    vals.push_back("0");
            keys.push_back("trt_engine_cache_enable");      vals.push_back("1");
            keys.push_back("trt_engine_cache_path");        vals.push_back(engine_cache_path.c_str());
            keys.push_back("trt_engine_cache_prefix");      vals.push_back("ot_rfdetr");
            // Reuse kernel auto-tuning results across builds (4-14x faster (re)builds)
            // and search harder for fast tactics (level 5 = max; only lengthens the
            // one-time build, which the cache then amortizes).
            keys.push_back("trt_timing_cache_enable");      vals.push_back("1");
            keys.push_back("trt_builder_optimization_level"); vals.push_back("5");
            // 8 GB workspace (the 3090 has 24 GB) so the builder isn't forced to
            // reject the fastest conv/attention tactics for lack of scratch space.
            keys.push_back("trt_max_workspace_size");       vals.push_back("8589934592");
            if (precision == "fp16" || precision == "int8") {
                keys.push_back("trt_fp16_enable");          vals.push_back("1");
            }
            // Dynamic-shape optimization profile: one engine for the whole batch
            // range [min,max] (e.g. 1 for the locked ROI .. 8 for SAHI tiles),
            // tuned for `opt`. Without this each new batch triggers a fresh
            // (minutes-long) engine build. Strings must outlive UpdateTRT below.
            auto shape_str = [&](int b) {
                return profile.input_name + ":" + std::to_string(b) + "x" +
                       std::to_string(profile.c) + "x" + std::to_string(profile.h) + "x" +
                       std::to_string(profile.w);
            };
            const std::string prof_min = profile.enabled() ? shape_str(profile.min_batch) : "";
            const std::string prof_opt = profile.enabled() ? shape_str(profile.opt_batch) : "";
            const std::string prof_max = profile.enabled() ? shape_str(profile.max_batch) : "";
            if (profile.enabled()) {
                keys.push_back("trt_profile_min_shapes"); vals.push_back(prof_min.c_str());
                keys.push_back("trt_profile_opt_shapes"); vals.push_back(prof_opt.c_str());
                keys.push_back("trt_profile_max_shapes"); vals.push_back(prof_max.c_str());
            }
            if (precision == "int8") {
                keys.push_back("trt_int8_enable");     vals.push_back("1");
                if (int8_calib_.empty()) {
                    throw std::runtime_error(
                        "TensorRT precision 'int8' needs a calibration table "
                        "(detector.int8_calib in the config)");
                }
                keys.push_back("trt_int8_calibration_table_name");
                vals.push_back(int8_calib_.c_str());
                keys.push_back("trt_int8_use_native_calibration_table");
                vals.push_back("1");
            }
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
                trt, keys.data(), vals.data(), keys.size()));
            opts_.AppendExecutionProvider_TensorRT_V2(*trt);
            name_ = "onnxruntime(tensorrt:" + precision + ")";
            trt_active = true;
            use_cuda = true;  // also append CUDA as the TRT fallback EP
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[ort] TensorRT EP unavailable (%s) — falling back to CUDA/CPU\n", e.what());
            use_cuda = true;
        }
    }

    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);  // throws if the provider can't load
            if (!trt_active) name_ = "onnxruntime(cuda)";
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr,
                "[ort] CUDA provider unavailable (%s) — falling back to CPU\n", e.what());
            if (!trt_active) name_ = "onnxruntime(cpu)";
        }
    }

    if (trt_active) {
        std::fprintf(stderr,
            "[ort] preparing TensorRT engine for '%s' — the FIRST run builds and caches it,\n"
            "      which can take several MINUTES (no window appears until this finishes).\n"
            "      Subsequent runs load the cached engine in seconds.\n",
            model_path.c_str());
    }
    try {
        session_ = Ort::Session(env_, model_path.c_str(), opts_);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("OrtBackend: failed to load '" + model_path + "': " + e.what());
    }
    const size_t n_in  = session_.GetInputCount();
    const size_t n_out = session_.GetOutputCount();
    if (n_in < 1 || n_out < 1) {
        throw std::runtime_error("OrtBackend: model has no inputs/outputs");
    }
    for (size_t i = 0; i < n_in; ++i)
        input_name_store_.push_back(session_.GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < n_out; ++i)
        output_name_store_.push_back(session_.GetOutputNameAllocated(i, alloc_).get());
    for (const auto& s : input_name_store_)  input_names_.push_back(s.c_str());
    for (const auto& s : output_name_store_) output_names_.push_back(s.c_str());
    std::fprintf(stderr, "[ort] backend = %s (%zu in, %zu out)\n", name_.c_str(), n_in, n_out);
}

std::vector<cv::Mat> OrtBackend::run_multi(const cv::Mat& blob_nchw) {
    CV_Assert(blob_nchw.dims == 4 && blob_nchw.type() == CV_32F && blob_nchw.isContinuous());
    const std::array<int64_t, 4> in_shape = {
        blob_nchw.size[0], blob_nchw.size[1], blob_nchw.size[2], blob_nchw.size[3]};
    const size_t in_count = static_cast<size_t>(
        in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3]);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(blob_nchw.ptr<float>()), in_count,
        in_shape.data(), in_shape.size());

    const auto t0 = std::chrono::steady_clock::now();
    auto outs = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &in, 1,
                             output_names_.data(), output_names_.size());
    if (profile_) {
        const auto t1 = std::chrono::steady_clock::now();
        infer_ms_sum_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (++infer_calls_ % 120 == 0) {
            std::fprintf(stderr, "[profile] %s: inference (Run incl. copies) avg %.2f ms"
                         " over %ld calls  (batch=%lld)\n",
                         name_.c_str(), infer_ms_sum_ / infer_calls_, infer_calls_,
                         static_cast<long long>(in_shape[0]));
        }
    }

    std::vector<cv::Mat> results;
    results.reserve(outs.size());
    for (auto& out : outs) {
        const auto info = out.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = info.GetShape();   // e.g. [1,300,4] or [1,16,8400]
        std::vector<int> sizes(shape.begin(), shape.end());
        const float* data = out.GetTensorData<float>();
        // Copy into an owned cv::Mat (the Ort::Value frees its buffer on return).
        cv::Mat m(static_cast<int>(sizes.size()), sizes.data(), CV_32F);
        std::memcpy(m.ptr<float>(), data, m.total() * sizeof(float));
        results.push_back(std::move(m));
    }
    return results;
}

cv::Mat OrtBackend::run(const cv::Mat& blob_nchw) {
    return std::move(run_multi(blob_nchw).front());
}

std::unique_ptr<InferenceBackend> make_backend(const std::string& kind,
                                               const std::string& model_path,
                                               const std::string& device,
                                               const std::string& precision,
                                               const std::string& int8_calib,
                                               const TrtProfile& profile) {
    const bool cuda = (device == "cuda");
    if (kind == "onnxruntime" || kind == "ort") {
        return std::make_unique<OrtBackend>(model_path, cuda, /*tensorrt=*/false,
                                            precision, int8_calib, profile);
    }
    if (kind == "tensorrt" || kind == "trt") {
        return std::make_unique<OrtBackend>(model_path, /*use_cuda=*/true, /*tensorrt=*/true,
                                            precision, int8_calib, profile);
    }
    if (kind == "ocv_dnn") {
        throw std::runtime_error(
            "make_backend: 'ocv_dnn' backend cannot run YOLOv8 on OpenCV " CV_VERSION
            " — use backend: onnxruntime");
    }
    throw std::runtime_error("make_backend: unknown backend '" + kind + "'");
}

}  // namespace ot
