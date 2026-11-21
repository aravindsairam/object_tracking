// Unit test for the YAML config loader (no GPU/model/video — pure logic).
//
// Guards two things:
//  1. Every shipped config in configs/ still loads (a strict schema must not
//     reject a key we actually use).
//  2. Unknown sections/keys are REJECTED. The loader used to ignore them, so a
//     config written for a build with `tracker.asso_func` loaded fine on a build
//     without it and silently ran plain IoU — a measured A/B became meaningless.
//
// Run from the repo root (it globs configs/*.yaml relative to the cwd):
//   ./build/test_config
#include "ot/config.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// Write `body` to a temp YAML and return its path.
std::string write_tmp(const std::string& name, const std::string& body) {
    const fs::path p = fs::temp_directory_path() / ("ot_test_config_" + name + ".yaml");
    std::ofstream(p) << body;
    return p.string();
}

// Load `body`; return the exception message, or "" if it loaded.
std::string load_error(const std::string& name, const std::string& body) {
    try {
        ot::load_config(write_tmp(name, body));
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

const char* kMinimal = "detector:\n  model_path: models/x.onnx\n";

void test_shipped_configs_load() {
    std::printf("test: every configs/*.yaml loads under the strict schema\n");
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator("configs"))
        if (e.path().extension() == ".yaml") files.push_back(e.path());
    check(!files.empty(), "found configs/*.yaml (run from the repo root)");
    for (const auto& f : files) {
        std::string err;
        try {
            ot::load_config(f.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        if (!err.empty()) std::printf("    %s: %s\n", f.string().c_str(), err.c_str());
        check(err.empty(), f.filename().string().c_str());
    }
}

void test_unknown_keys_rejected() {
    std::printf("test: unknown keys and sections are rejected\n");
    const std::string e1 = load_error("typo", std::string(kMinimal) +
                                      "tracker:\n  type: ocsort\n  asso_fucn: diou\n");
    check(e1.find("asso_fucn") != std::string::npos && e1.find("tracker") != std::string::npos,
          "misspelt tracker key names the key and the section");

    const std::string e2 = load_error("section", std::string(kMinimal) + "trackr:\n  type: ocsort\n");
    check(e2.find("trackr") != std::string::npos, "misspelt top-level section is rejected");

    const std::string e3 = load_error("lockkey", std::string(kMinimal) + "lock:\n  reacquire_tresh: 0.8\n");
    check(e3.find("reacquire_tresh") != std::string::npos, "misspelt lock key is rejected");

    const std::string e4 = load_error("sizeprior", std::string(kMinimal) +
                                      "range:\n  size_priors:\n    tractor: 5.0\n");
    check(e4.empty(), "free-form range.size_priors class names are still allowed");
}

void test_tracker_association_keys() {
    std::printf("test: asso_func / buffer_ratio / reacquire_thresh_single parse\n");
    const ot::Config d = ot::load_config("configs/rfdetr_medium_sahi_ocsort_u8_diou.yaml");
    check(d.tracker.asso_func == "diou", "diou config -> asso_func diou");
    check(d.tracker.iou_threshold > 0.39f && d.tracker.iou_threshold < 0.41f,
          "diou config -> iou_threshold 0.4");

    const ot::Config b = ot::load_config("configs/rfdetr_medium_sahi_ocsort_u8_biou.yaml");
    check(b.tracker.asso_func == "biou", "biou config -> asso_func biou");
    check(b.tracker.buffer_ratio > 0.49f && b.tracker.buffer_ratio < 0.51f,
          "biou config -> buffer_ratio 0.5");

    const ot::Config base = ot::load_config("configs/rfdetr_medium_sahi_ocsort_u8.yaml");
    check(base.tracker.asso_func == "iou", "base config keeps plain iou (the A/B baseline)");
    check(base.lock.reacquire_thresh_single > 0.89f && base.lock.reacquire_thresh_single < 0.91f,
          "base config -> reacquire_thresh_single 0.90");

    const std::string bad = load_error("badasso", std::string(kMinimal) +
                                       "tracker:\n  type: ocsort\n  asso_func: dIoU\n");
    check(bad.find("asso_func") != std::string::npos, "unknown asso_func value is rejected");
}

}  // namespace

int main() {
    test_shipped_configs_load();
    test_unknown_keys_rejected();
    test_tracker_association_keys();
    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
