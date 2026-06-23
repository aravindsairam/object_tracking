// Model-free ReID embedder: a normalized HSV hue-saturation histogram of the
// box crop. Needs no download or GPU and is weaker than the neural OSNet
// embedder, but gives a usable appearance signal for lock re-acquisition.
// See onnx_reid_embedder.cpp for the stronger neural alternative.
#include "ot/reid.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ot {

float ReidEmbedder::cosine(const Embedding& a, const Embedding& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;  // inputs are L2-normalized
}

namespace {
// HSV Hue-Saturation 2D histogram, L2-normalized. Robust-ish to brightness;
// captures the dominant color of a small target.
constexpr int kHBins = 16;
constexpr int kSBins = 8;

class HistogramEmbedder : public ReidEmbedder {
public:
    Embedding embed(const cv::Mat& frame, const BBox& box) override {
        cv::Rect r = box.to_rect() & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width < 2 || r.height < 2) return {};

        cv::Mat hsv;
        cv::cvtColor(frame(r), hsv, cv::COLOR_BGR2HSV);

        const int hist_size[] = {kHBins, kSBins};
        const float h_range[] = {0, 180};
        const float s_range[] = {0, 256};
        const float* ranges[] = {h_range, s_range};
        const int channels[] = {0, 1};

        cv::Mat hist;
        cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges);

        Embedding e(kHBins * kSBins);
        float norm = 0.0f;
        for (int i = 0; i < kHBins * kSBins; ++i) {
            const float v = hist.at<float>(i / kSBins, i % kSBins);
            e[i] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-6f) for (float& v : e) v /= norm;
        return e;
    }
};
}  // namespace

std::shared_ptr<ReidEmbedder> make_histogram_embedder() {
    return std::make_shared<HistogramEmbedder>();
}

}  // namespace ot
