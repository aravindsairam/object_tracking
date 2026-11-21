// YAML -> Config loader. One flat function reads each section (detector /
// tiling / tracker / lock / reid / input / output / range) via get_or, so any
// missing key falls back to the default already set on the Config struct.
// Unknown sections and keys are REJECTED (a typo or a key this build does not
// understand must not load silently and run the defaults). Beyond that only the
// few hard requirements are validated (model paths present, enums known).
// Throws on a missing/unparseable file.
#include "ot/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ot {

namespace {
// Read node[key] as T if present, else keep `fallback`.
template <typename T>
T get_or(const YAML::Node& node, const char* key, const T& fallback) {
    if (node && node[key]) return node[key].as<T>();
    return fallback;
}

// Throw if `node` (a YAML map, or absent) has any key outside `allowed`.
// `where` names the section in the error ("tracker", or "<top level>").
void require_known_keys(const YAML::Node& node, const std::string& where,
                        std::initializer_list<const char*> allowed,
                        const std::string& yaml_path) {
    if (!node) return;
    if (!node.IsMap()) {
        throw std::runtime_error("load_config: '" + where + "' in " + yaml_path +
                                 " must be a mapping");
    }
    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        const bool known = std::any_of(allowed.begin(), allowed.end(),
                                       [&](const char* a) { return key == a; });
        if (known) continue;
        std::string list;
        for (const char* a : allowed) list += std::string(list.empty() ? "" : ", ") + a;
        throw std::runtime_error("load_config: unknown key '" + key + "' in " + where +
                                 " of " + yaml_path + " (known: " + list + ")");
    }
}
}  // namespace

Config load_config(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("load_config: cannot read '" + yaml_path +
                                 "': " + e.what());
    }

    // Strict schema: every section and key this loader reads, and nothing else.
    // Keep these lists in sync with the get_or calls below.
    require_known_keys(root, "<top level>",
                       {"detector", "tiling", "tracker", "lock", "reid", "output", "range", "input"},
                       yaml_path);
    require_known_keys(root["detector"], "detector",
                       {"family", "model_path", "backend", "device", "precision", "int8_calib",
                        "input_size", "input_u8", "b1_session", "conf", "nms_iou", "class_map"},
                       yaml_path);
    require_known_keys(root["tiling"], "tiling", {"enabled", "tile", "overlap", "full_frame"},
                       yaml_path);
    require_known_keys(root["tracker"], "tracker",
                       {"type", "track_thresh", "match_thresh", "track_buffer", "min_conf",
                        "iou_threshold", "asso_func", "buffer_ratio", "new_track_thresh",
                        "proximity_thresh", "appearance_thresh", "cmc_method", "with_reid",
                        "max_age", "delta_t", "inertia", "use_byte", "q_xy_scaling", "q_s_scaling"},
                       yaml_path);
    require_known_keys(root["lock"], "lock",
                       {"coast_to_lost", "reacquire_thresh", "smoothing", "verify_thresh",
                        "reacquire_max_frac", "reacquire_margin", "reacquire_thresh_single",
                        "roi_fastpath", "roi_full_interval", "roi_scale"},
                       yaml_path);
    require_known_keys(root["reid"], "reid",
                       {"kind", "model_path", "backend", "device", "precision", "input_w", "input_h"},
                       yaml_path);
    require_known_keys(root["output"], "output", {"sink", "path"}, yaml_path);
    require_known_keys(root["range"], "range",
                       {"enabled", "smooth_window", "max_range_m", "min_depression_deg",
                        "ground_alt_offset_m", "equiv_focal_mm", "size_priors"},
                       yaml_path);
    require_known_keys(root["input"], "input", {"height", "decoder"}, yaml_path);

    Config cfg;
    const YAML::Node d = root["detector"];
    cfg.detector.family     = get_or<std::string>(d, "family", cfg.detector.family);
    cfg.detector.model_path = get_or<std::string>(d, "model_path", cfg.detector.model_path);
    cfg.detector.backend    = get_or<std::string>(d, "backend", cfg.detector.backend);
    cfg.detector.device     = get_or<std::string>(d, "device", cfg.detector.device);
    cfg.detector.precision  = get_or<std::string>(d, "precision", cfg.detector.precision);
    cfg.detector.int8_calib = get_or<std::string>(d, "int8_calib", cfg.detector.int8_calib);
    cfg.detector.input_size = get_or<int>(d, "input_size", cfg.detector.input_size);
    cfg.detector.input_u8   = get_or<bool>(d, "input_u8", cfg.detector.input_u8);
    cfg.detector.b1_session = get_or<bool>(d, "b1_session", cfg.detector.b1_session);
    cfg.detector.conf       = get_or<float>(d, "conf", cfg.detector.conf);
    cfg.detector.nms_iou    = get_or<float>(d, "nms_iou", cfg.detector.nms_iou);
    cfg.detector.class_map  = get_or<std::string>(d, "class_map", cfg.detector.class_map);

    const YAML::Node t = root["tiling"];
    cfg.tiling.enabled    = get_or<bool>(t, "enabled", cfg.tiling.enabled);
    cfg.tiling.tile       = get_or<int>(t, "tile", cfg.tiling.tile);
    cfg.tiling.overlap    = get_or<float>(t, "overlap", cfg.tiling.overlap);
    cfg.tiling.full_frame = get_or<bool>(t, "full_frame", cfg.tiling.full_frame);

    const YAML::Node tr = root["tracker"];
    cfg.tracker.type             = get_or<std::string>(tr, "type", cfg.tracker.type);
    cfg.tracker.track_thresh     = get_or<float>(tr, "track_thresh", cfg.tracker.track_thresh);
    cfg.tracker.match_thresh     = get_or<float>(tr, "match_thresh", cfg.tracker.match_thresh);
    cfg.tracker.track_buffer     = get_or<int>(tr, "track_buffer", cfg.tracker.track_buffer);
    cfg.tracker.min_conf         = get_or<float>(tr, "min_conf", cfg.tracker.min_conf);
    cfg.tracker.iou_threshold    = get_or<float>(tr, "iou_threshold", cfg.tracker.iou_threshold);
    cfg.tracker.asso_func        = get_or<std::string>(tr, "asso_func", cfg.tracker.asso_func);
    cfg.tracker.buffer_ratio     = get_or<float>(tr, "buffer_ratio", cfg.tracker.buffer_ratio);
    cfg.tracker.new_track_thresh = get_or<float>(tr, "new_track_thresh", cfg.tracker.new_track_thresh);
    cfg.tracker.proximity_thresh = get_or<float>(tr, "proximity_thresh", cfg.tracker.proximity_thresh);
    cfg.tracker.appearance_thresh = get_or<float>(tr, "appearance_thresh", cfg.tracker.appearance_thresh);
    cfg.tracker.cmc_method       = get_or<std::string>(tr, "cmc_method", cfg.tracker.cmc_method);
    cfg.tracker.with_reid        = get_or<bool>(tr, "with_reid", cfg.tracker.with_reid);
    cfg.tracker.max_age          = get_or<int>(tr, "max_age", cfg.tracker.max_age);
    cfg.tracker.delta_t          = get_or<int>(tr, "delta_t", cfg.tracker.delta_t);
    cfg.tracker.inertia          = get_or<float>(tr, "inertia", cfg.tracker.inertia);
    cfg.tracker.use_byte         = get_or<bool>(tr, "use_byte", cfg.tracker.use_byte);
    cfg.tracker.q_xy_scaling     = get_or<float>(tr, "q_xy_scaling", cfg.tracker.q_xy_scaling);
    cfg.tracker.q_s_scaling      = get_or<float>(tr, "q_s_scaling", cfg.tracker.q_s_scaling);
    if (cfg.tracker.type != "bytetrack" && cfg.tracker.type != "botsort" &&
        cfg.tracker.type != "ocsort") {
        throw std::runtime_error("load_config: tracker.type must be 'bytetrack', 'botsort' or "
                                 "'ocsort', got '" + cfg.tracker.type + "'");
    }
    if (cfg.tracker.asso_func != "iou" && cfg.tracker.asso_func != "giou" &&
        cfg.tracker.asso_func != "diou" && cfg.tracker.asso_func != "ciou" &&
        cfg.tracker.asso_func != "hmiou" && cfg.tracker.asso_func != "biou") {
        throw std::runtime_error("load_config: tracker.asso_func must be one of 'iou', 'giou', "
                                 "'diou', 'ciou', 'hmiou', 'biou', got '" + cfg.tracker.asso_func + "'");
    }

    const YAML::Node l = root["lock"];
    cfg.lock.coast_to_lost    = get_or<int>(l, "coast_to_lost", cfg.lock.coast_to_lost);
    cfg.lock.reacquire_thresh = get_or<float>(l, "reacquire_thresh", cfg.lock.reacquire_thresh);
    cfg.lock.smoothing        = get_or<float>(l, "smoothing", cfg.lock.smoothing);
    cfg.lock.verify_thresh    = get_or<float>(l, "verify_thresh", cfg.lock.verify_thresh);
    cfg.lock.reacquire_max_frac = get_or<float>(l, "reacquire_max_frac", cfg.lock.reacquire_max_frac);
    cfg.lock.reacquire_margin = get_or<float>(l, "reacquire_margin", cfg.lock.reacquire_margin);
    cfg.lock.reacquire_thresh_single = get_or<float>(l, "reacquire_thresh_single", cfg.lock.reacquire_thresh_single);
    cfg.lock.roi_fastpath     = get_or<bool>(l, "roi_fastpath", cfg.lock.roi_fastpath);
    cfg.lock.roi_full_interval = get_or<int>(l, "roi_full_interval", cfg.lock.roi_full_interval);
    cfg.lock.roi_scale        = get_or<float>(l, "roi_scale", cfg.lock.roi_scale);

    const YAML::Node rd = root["reid"];
    cfg.reid.kind       = get_or<std::string>(rd, "kind", cfg.reid.kind);
    cfg.reid.model_path = get_or<std::string>(rd, "model_path", cfg.reid.model_path);
    cfg.reid.backend    = get_or<std::string>(rd, "backend", cfg.reid.backend);
    cfg.reid.device     = get_or<std::string>(rd, "device", cfg.reid.device);
    cfg.reid.precision  = get_or<std::string>(rd, "precision", cfg.reid.precision);
    cfg.reid.input_w    = get_or<int>(rd, "input_w", cfg.reid.input_w);
    cfg.reid.input_h    = get_or<int>(rd, "input_h", cfg.reid.input_h);
    if (cfg.reid.kind == "onnx" && cfg.reid.model_path.empty()) {
        throw std::runtime_error("load_config: reid.model_path is required when reid.kind: onnx");
    }

    const YAML::Node o = root["output"];
    cfg.output.sink = get_or<std::string>(o, "sink", cfg.output.sink);
    cfg.output.path = get_or<std::string>(o, "path", cfg.output.path);

    const YAML::Node rg = root["range"];
    cfg.range.enabled             = get_or<bool>(rg, "enabled", cfg.range.enabled);
    cfg.range.smooth_window       = get_or<int>(rg, "smooth_window", cfg.range.smooth_window);
    cfg.range.max_range_m         = get_or<float>(rg, "max_range_m", cfg.range.max_range_m);
    cfg.range.min_depression_deg  = get_or<float>(rg, "min_depression_deg", cfg.range.min_depression_deg);
    cfg.range.ground_alt_offset_m = get_or<float>(rg, "ground_alt_offset_m", cfg.range.ground_alt_offset_m);
    cfg.range.equiv_focal_mm      = get_or<float>(rg, "equiv_focal_mm", cfg.range.equiv_focal_mm);
    if (rg && rg["size_priors"]) {  // merged over the defaults, not replacing them
        for (const auto& kv : rg["size_priors"])
            cfg.range.size_priors[kv.first.as<std::string>()] = kv.second.as<float>();
    }

    cfg.work_height = get_or<int>(root["input"], "height", cfg.work_height);
    cfg.decoder     = get_or<std::string>(root["input"], "decoder", cfg.decoder);

    if (cfg.detector.model_path.empty()) {
        throw std::runtime_error("load_config: detector.model_path is required in " + yaml_path);
    }
    return cfg;
}

}  // namespace ot
