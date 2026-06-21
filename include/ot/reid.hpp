#pragma once

#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace ot {

// Produces an appearance signature for a box, used to re-identify the locked
// target after it is lost. Kept behind an interface so the cheap histogram
// embedder (good for small aerial targets) can be swapped for a neural one.
class ReidEmbedder {
public:
    virtual ~ReidEmbedder() = default;

    // Returns an L2-normalized appearance feature for `box` within `frame`.
    // Returns an empty vector if the box is degenerate.
    virtual Embedding embed(const cv::Mat& frame, const BBox& box) = 0;

    // Cosine similarity of two L2-normalized embeddings (0 if either is empty).
    static float cosine(const Embedding& a, const Embedding& b);
};

// HSV color-histogram embedder: lightweight, model-free, effective for small
// aerial vehicles/people where neural person-ReID features are unreliable.
std::shared_ptr<ReidEmbedder> make_histogram_embedder();

// Neural ReID embedder: runs an OSNet-style ReID ONNX (input [1,3,input_h,input_w]
// ImageNet-normalized RGB, output a [1,D] feature) on the shared OrtBackend, then
// L2-normalizes. `backend` is "onnxruntime" | "tensorrt"; `precision` applies to
// the TensorRT EP only. Stronger than the histogram at telling apart same-colored
// targets, at the cost of one small inference per embed.
std::shared_ptr<ReidEmbedder> make_onnx_reid_embedder(
    const std::string& model_path, int input_w, int input_h,
    const std::string& backend, const std::string& device, const std::string& precision);

}  // namespace ot
