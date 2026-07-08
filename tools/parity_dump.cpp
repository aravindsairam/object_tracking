// Offline detection dumper for the backend-parity / accuracy audit. Two modes:
//
//   video: run the FULL production path (SAHI tiling + IOS merge, real factory)
//          over every Nth frame of a clip and dump merged detections as JSONL.
//          Run it once per backend (--backend/--precision override the config)
//          and diff the dumps with tools/parity_compare.py — same decode path
//          every run, so frames are pixel-identical across backends.
//
//   coco:  run batch-1 raw inference over a directory of COCO images and dump
//          RF-DETR's OFFICIAL postprocess (top-K over all Q*C sigmoid scores,
//          class id = flat % C, i.e. the 91-slot COCO category id) so the mAP
//          from tools/coco_eval_dump.py is comparable to the paper's number.
//
//   ./build/parity_dump video <clip> <config.yaml> --out d.jsonl [--every 30]
//        [--max 200] [--backend tensorrt|onnxruntime] [--precision fp16|fp32]
//        [--conf 0.1]
//   ./build/parity_dump coco <img_dir> <config.yaml> --out d.jsonl
//        [--backend ...] [--precision ...] [--topk 300] [--limit 0]
#include "ot/config.hpp"
#include "ot/detector_factory.hpp"
#include "rfdetr_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string mode, source, config, out;
    std::string backend, precision;      // empty = keep config value
    int   every = 30, max_frames = 200;  // video mode
    int   topk = 300, limit = 0;         // coco mode (limit 0 = all images)
    float conf = -1.0f;                  // <0 = keep config value
};

bool parse(int argc, char** argv, Args& a) {
    if (argc < 4) return false;
    a.mode = argv[1]; a.source = argv[2]; a.config = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (k == "--out")       { if (const char* v = next()) a.out = v; }
        else if (k == "--every")     { if (const char* v = next()) a.every = std::atoi(v); }
        else if (k == "--max")       { if (const char* v = next()) a.max_frames = std::atoi(v); }
        else if (k == "--topk")      { if (const char* v = next()) a.topk = std::atoi(v); }
        else if (k == "--limit")     { if (const char* v = next()) a.limit = std::atoi(v); }
        else if (k == "--backend")   { if (const char* v = next()) a.backend = v; }
        else if (k == "--precision") { if (const char* v = next()) a.precision = v; }
        else if (k == "--conf")      { if (const char* v = next()) a.conf = std::atof(v); }
        else { std::fprintf(stderr, "unknown arg '%s'\n", k.c_str()); return false; }
    }
    return (a.mode == "video" || a.mode == "coco") && !a.out.empty();
}

int run_video(const Args& a, ot::Config cfg) {
    auto det = ot::make_detector(cfg.detector, cfg.tiling);

    cv::VideoCapture cap(a.source);
    if (!cap.isOpened()) { std::fprintf(stderr, "cannot open '%s'\n", a.source.c_str()); return 1; }
    std::FILE* out = std::fopen(a.out.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write '%s'\n", a.out.c_str()); return 1; }

    cv::Mat frame, work;
    int idx = -1, done = 0;
    while (done < a.max_frames && cap.grab()) {
        ++idx;
        if (idx % a.every != 0) continue;   // grab() skips the BGR conversion
        if (!cap.retrieve(frame)) break;
        // Match the app/smoke tool: integer-math downscale to the working height.
        if (cfg.work_height > 0 && frame.rows > cfg.work_height) {
            const int w = frame.cols * cfg.work_height / frame.rows;
            cv::resize(frame, work, cv::Size(w, cfg.work_height), 0, 0, cv::INTER_AREA);
        } else {
            work = frame;
        }
        const auto dets = det->detect(work);
        std::fprintf(out, "{\"frame\":%d,\"w\":%d,\"h\":%d,\"dets\":[", idx, work.cols, work.rows);
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto& d = dets[i];
            std::fprintf(out, "%s{\"c\":%d,\"s\":%.5f,\"b\":[%.2f,%.2f,%.2f,%.2f]}",
                         i ? "," : "", d.class_id, d.score,
                         d.box.x, d.box.y, d.box.w, d.box.h);
        }
        std::fprintf(out, "]}\n");
        if (++done % 50 == 0) std::fprintf(stderr, "  %d frames dumped\n", done);
    }
    std::fclose(out);
    std::fprintf(stderr, "video dump: %d frames -> %s\n", done, a.out.c_str());
    return 0;
}

int run_coco(const Args& a, ot::Config cfg) {
    if (cfg.detector.family != "rfdetr") {
        std::fprintf(stderr, "coco mode supports family=rfdetr only\n"); return 1;
    }
    ot::RfDetrDetector det(cfg.detector);

    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::directory_iterator(a.source)) {
        if (e.path().extension() == ".jpg") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (a.limit > 0 && static_cast<int>(files.size()) > a.limit) files.resize(a.limit);

    std::FILE* out = std::fopen(a.out.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write '%s'\n", a.out.c_str()); return 1; }

    int done = 0;
    for (const auto& f : files) {
        const cv::Mat img = cv::imread(f.string(), cv::IMREAD_COLOR);
        if (img.empty()) { std::fprintf(stderr, "skip unreadable %s\n", f.c_str()); continue; }
        const long image_id = std::atol(f.stem().string().c_str());

        auto [boxes, logits] = det.run_raw({img});
        const int Q = boxes.size[boxes.dims - 2];
        const int C = logits.size[logits.dims - 1];
        const float* b = boxes.ptr<float>();
        const float* l = logits.ptr<float>();

        // Official RF-DETR postprocess: top-K over the flattened Q*C scores
        // (sigmoid is monotonic, so rank by raw logit and sigmoid only the top-K).
        std::vector<int> order(static_cast<size_t>(Q) * C);
        std::iota(order.begin(), order.end(), 0);
        const int K = std::min<int>(a.topk, static_cast<int>(order.size()));
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
                          [&](int x, int y) { return l[x] > l[y]; });

        std::fprintf(out, "{\"image_id\":%ld,\"w\":%d,\"h\":%d,\"dets\":[",
                     image_id, img.cols, img.rows);
        for (int i = 0; i < K; ++i) {
            const int flat = order[i];
            const int q = flat / C, c = flat % C;
            const float score = 1.0f / (1.0f + std::exp(-l[flat]));
            const float* bb = b + static_cast<size_t>(q) * 4;
            const float w  = bb[2] * img.cols, h = bb[3] * img.rows;
            const float x  = bb[0] * img.cols - w * 0.5f;
            const float y  = bb[1] * img.rows - h * 0.5f;
            std::fprintf(out, "%s[%d,%.5f,%.2f,%.2f,%.2f,%.2f]",
                         i ? "," : "", c, score, x, y, w, h);
        }
        std::fprintf(out, "]}\n");
        if (++done % 500 == 0) std::fprintf(stderr, "  %d/%zu images\n", done, files.size());
    }
    std::fclose(out);
    std::fprintf(stderr, "coco dump: %d images -> %s\n", done, a.out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr,
            "usage: %s video <clip>    <config.yaml> --out d.jsonl [--every N] [--max M]\n"
            "       %s coco  <img_dir> <config.yaml> --out d.jsonl [--topk K] [--limit L]\n"
            "       common: [--backend tensorrt|onnxruntime] [--precision fp16|fp32] [--conf C]\n",
            argv[0], argv[0]);
        return 2;
    }
    ot::Config cfg = ot::load_config(a.config);
    if (!a.backend.empty())   cfg.detector.backend   = a.backend;
    if (!a.precision.empty()) cfg.detector.precision = a.precision;
    if (a.conf >= 0.0f)       cfg.detector.conf      = a.conf;
    std::fprintf(stderr, "[parity_dump] mode=%s backend=%s precision=%s conf=%.2f\n",
                 a.mode.c_str(), cfg.detector.backend.c_str(),
                 cfg.detector.precision.c_str(), cfg.detector.conf);
    return a.mode == "video" ? run_video(a, std::move(cfg)) : run_coco(a, std::move(cfg));
}
