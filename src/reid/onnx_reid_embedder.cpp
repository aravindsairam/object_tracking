#include "ot/reid.hpp"
#include "ot/inference_backend.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ot {

namespace {
// torchreid / OSNet preprocessing: ImageNet mean/std on RGB in [0,1].
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

// Runs a ReID ONNX over a box crop and returns an L2-normalized feature.
class OnnxReidEmbedder : public ReidEmbedder {
public:
    OnnxReidEmbedder(const std::string& model_path, int input_w, int input_h,
                     const std::string& backend, const std::string& device,
                     const std::string& precision)
        : in_w_(input_w), in_h_(input_h),
          backend_(make_backend(backend, model_path, device, precision)) {}

    Embedding embed(const cv::Mat& frame, const BBox& box) override {
        const cv::Rect r = box.to_rect() & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width < 2 || r.height < 2) return {};

        cv::Mat resized, rgb;
        cv::resize(frame(r), resized, cv::Size(in_w_, in_h_), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        const int sizes[4] = {1, 3, in_h_, in_w_};
        cv::Mat blob(4, sizes, CV_32F);
        cv::Mat chan;
        for (int c = 0; c < 3; ++c) {  // c: 0=R, 1=G, 2=B
            cv::Mat plane(in_h_, in_w_, CV_32F,
                          blob.ptr<float>() + static_cast<size_t>(c) * in_h_ * in_w_);
            cv::extractChannel(rgb, chan, c);
            chan.convertTo(plane, CV_32F, 1.0 / 255.0);
            plane = (plane - kMean[c]) / kStd[c];
        }

        const cv::Mat out = backend_->run(blob);   // [1, D] feature
        const int D = static_cast<int>(out.total());
        const float* p = out.ptr<float>();

        Embedding e(D);
        float norm = 0.0f;
        for (int i = 0; i < D; ++i) { e[i] = p[i]; norm += p[i] * p[i]; }
        norm = std::sqrt(norm);
        if (norm > 1e-6f) for (float& v : e) v /= norm;
        return e;
    }

private:
    int in_w_, in_h_;
    std::unique_ptr<InferenceBackend> backend_;
};
}  // namespace

std::shared_ptr<ReidEmbedder> make_onnx_reid_embedder(
    const std::string& model_path, int input_w, int input_h,
    const std::string& backend, const std::string& device, const std::string& precision) {
    return std::make_shared<OnnxReidEmbedder>(model_path, input_w, input_h,
                                              backend, device, precision);
}

}  // namespace ot
