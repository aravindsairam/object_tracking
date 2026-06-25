// YAML -> Config loader. One flat function reads each section (detector /
// tiling / tracker / lock / reid / input / output) via get_or, so any missing
// key falls back to the default already set on the Config struct. Only the few
// hard requirements are validated (model paths present, tracker.type known);
// everything else is optional. Throws on a missing/unparseable file.
#include "ot/config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace ot {

namespace {
// Read node[key] as T if present, else keep `fallback`.
template <typename T>
T get_or(const YAML::Node& node, const char* key, const T& fallback) {
    if (node && node[key]) return node[key].as<T>();
    return fallback;
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

    Config cfg;
    const YAML::Node d = root["detector"];
    cfg.detector.family     = get_or<std::string>(d, "family", cfg.detector.family);
    cfg.detector.model_path = get_or<std::string>(d, "model_path", cfg.detector.model_path);
    cfg.detector.backend    = get_or<std::string>(d, "backend", cfg.detector.backend);
    cfg.detector.device     = get_or<std::string>(d, "device", cfg.detector.device);
    cfg.detector.precision  = get_or<std::string>(d, "precision", cfg.detector.precision);
    cfg.detector.int8_calib = get_or<std::string>(d, "int8_calib", cfg.detector.int8_calib);
    cfg.detector.input_size = get_or<int>(d, "input_size", cfg.detector.input_size);
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

    const YAML::Node l = root["lock"];
    cfg.lock.coast_to_lost    = get_or<int>(l, "coast_to_lost", cfg.lock.coast_to_lost);
    cfg.lock.reacquire_thresh = get_or<float>(l, "reacquire_thresh", cfg.lock.reacquire_thresh);
    cfg.lock.smoothing        = get_or<float>(l, "smoothing", cfg.lock.smoothing);
    cfg.lock.verify_thresh    = get_or<float>(l, "verify_thresh", cfg.lock.verify_thresh);
    cfg.lock.reacquire_max_frac = get_or<float>(l, "reacquire_max_frac", cfg.lock.reacquire_max_frac);
    cfg.lock.reacquire_margin = get_or<float>(l, "reacquire_margin", cfg.lock.reacquire_margin);
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

    cfg.work_height = get_or<int>(root["input"], "height", cfg.work_height);
    cfg.decoder     = get_or<std::string>(root["input"], "decoder", cfg.decoder);

    if (cfg.detector.model_path.empty()) {
        throw std::runtime_error("load_config: detector.model_path is required in " + yaml_path);
    }
    return cfg;
}

}  // namespace ot
