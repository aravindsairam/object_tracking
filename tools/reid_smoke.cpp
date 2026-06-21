// Headless check for the neural ReID embedder. Builds the ONNX embedder from a
// config, embeds two boxes from one frame, and prints the feature dim plus a few
// cosine similarities (self-cosine must be ~1.0; an L2-normalized feature has
// cosine in [-1,1]). Verifies the embed -> backend -> L2-norm -> cosine path.
//
//   ./build/reid_smoke <video> <config.yaml> [frame_index]
#include "ot/config.hpp"
#include "ot/reid.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <video> <config.yaml> [frame_index]\n", argv[0]);
        return 2;
    }
    const ot::Config cfg = ot::load_config(argv[2]);
    auto reid = cfg.reid.kind == "onnx"
        ? ot::make_onnx_reid_embedder(cfg.reid.model_path, cfg.reid.input_w, cfg.reid.input_h,
                                      cfg.reid.backend, cfg.reid.device, cfg.reid.precision)
        : ot::make_histogram_embedder();
    std::printf("reid kind = %s\n", cfg.reid.kind.c_str());

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::fprintf(stderr, "cannot open video '%s'\n", argv[1]); return 1; }
    cap.set(cv::CAP_PROP_POS_FRAMES, (argc > 3) ? std::atoi(argv[3]) : 300);
    cv::Mat frame;
    if (!cap.read(frame)) { std::fprintf(stderr, "cannot read frame\n"); return 1; }

    // Two boxes: A and B (different region), plus A again to check determinism.
    const int W = frame.cols, H = frame.rows;
    ot::BBox a{W * 0.30f, H * 0.40f, W * 0.12f, H * 0.20f};
    ot::BBox b{W * 0.60f, H * 0.55f, W * 0.12f, H * 0.20f};

    const auto ea  = reid->embed(frame, a);
    const auto ea2 = reid->embed(frame, a);
    const auto eb  = reid->embed(frame, b);
    if (ea.empty() || eb.empty()) { std::fprintf(stderr, "empty embedding\n"); return 1; }

    std::printf("feature dim = %zu\n", ea.size());
    std::printf("cosine(A,A) = %.4f  (expect ~1.0)\n", ot::ReidEmbedder::cosine(ea, ea2));
    std::printf("cosine(A,B) = %.4f  (different regions -> usually < 1.0)\n",
                ot::ReidEmbedder::cosine(ea, eb));
    return 0;
}
