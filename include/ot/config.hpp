#pragma once

#include <string>

namespace ot {

// How to build and run the detector. Everything here is set from YAML so the
// model can be swapped without recompiling.
struct DetectorCfg {
    std::string family     = "yolo";   // yolo | waldo | unidrone | rfdetr (decode family)
    std::string model_path = "";       // path to the .onnx (or engine, later)
    std::string backend    = "onnxruntime";  // onnxruntime | tensorrt | ocv_dnn
    std::string device     = "cuda";    // cuda | cpu (onnxruntime execution provider)
    std::string precision  = "fp32";    // fp32 | fp16 | int8 (TensorRT EP only)
    std::string int8_calib = "";        // INT8 calibration table path (precision: int8 only)
    int         input_size = 640;       // square network input
    float       conf       = 0.25f;     // score threshold
    float       nms_iou    = 0.45f;     // NMS IoU (raw-head models only)
    std::string class_map  = "coco";    // coco | waldo | unidrone — which ids are people/vehicles
};

// SAHI-style tiled inference for small objects: slice the frame into overlapping
// tiles, detect each, merge. Best paired with a higher work_height.
struct TilingCfg {
    bool  enabled    = false;
    int   tile       = 640;    // tile size (px); match detector input_size
    float overlap    = 0.2f;   // fractional overlap between adjacent tiles
    bool  full_frame = true;   // also run one whole-frame pass (catches large objects)
};

// Multi-object tracker selection and association tuning. The detector feeds the
// same boxes to whichever tracker is chosen here, so switching is purely a config
// change (no recompile). ByteTrack is IoU + motion only (fast, the default).
// BoT-SORT adds camera-motion compensation (GMC) and appearance (ReID) fusion to
// the association — costlier per frame, but far steadier under camera motion and
// when similar-looking targets cross, which is the aerial failure mode.
struct TrackerCfg {
    std::string type = "bytetrack";   // bytetrack | botsort | ocsort

    // Shared association knobs (same meaning for both trackers).
    float track_thresh = 0.4f;   // high-confidence cut: above it a det can start/extend a
                                 // track in the first association pass (BoT-SORT: track_high_thresh).
    float match_thresh = 0.8f;   // max association cost (1-IoU, or fused) to accept a match.
    int   track_buffer = 30;     // frames a lost track is retained before deletion
                                 // (scaled by frame_rate/30 internally).
    float min_conf     = 0.1f;   // low-confidence floor: dets between this and track_thresh feed
                                 // the second association pass (BoT-SORT: track_low_thresh).

    // BoT-SORT only. Ignored when type: bytetrack.
    float new_track_thresh  = 0.6f;   // min confidence to spawn a brand-new id.
    float proximity_thresh  = 0.5f;   // IoU gate: appearance is only trusted for box pairs that
                                      // already overlap enough (rejects far look-alikes).
    float appearance_thresh = 0.25f;  // max embedding distance for appearance to fuse into the cost.
    std::string cmc_method  = "ecc";  // camera-motion compensation: "ecc" (OpenCV ECC) enables GMC;
                                      // any other value (e.g. "none") disables it.
    bool  with_reid         = true;   // fuse ReID appearance into association. Reuses the same
                                      // embedder as the lock (see ReidCfg); costs one embed per
                                      // detection per frame. Set false for motion + GMC only.

    // OC-SORT only. Ignored unless type: ocsort. Motion-only tracker (no ReID, no
    // GMC, no per-detection embedding cost) with an observation-centric motion model
    // that handles erratic/non-linear motion better than ByteTrack's linear Kalman —
    // the cheaper, more aerial-appropriate upgrade. Uses track_thresh (high-conf cut)
    // and min_conf (low floor) from the shared knobs above.
    int   delta_t  = 3;       // observation window (frames) for the velocity-direction estimate.
    float inertia  = 0.2f;    // weight of velocity-direction consistency in the cost (OCM term);
                              // higher trusts smooth motion more, lower lets IoU dominate.
    bool  use_byte = true;    // also run a ByteTrack-style low-conf second association pass — keeps
                              // flickering small targets (recommended for aerial small objects).
};

// Single-target lock behavior.
struct LockCfg {
    int   coast_to_lost = 30;          // missing frames tolerated before declaring LOST
    float reacquire_thresh = 0.6f;     // appearance cosine needed to auto re-lock

    // Per-frame appearance gate on the id-matched track: ByteTrack reuses a track
    // id across different objects, so a surviving id is not proof of identity. A
    // matched track whose appearance cosine to the template falls below this is
    // treated as a miss (coast) instead of a confident lock — the guard against
    // silently following an id-switch. Set <= 0 to disable (pure id-continuity).
    float verify_thresh = 0.35f;

    // Caps the appearance re-acquire search radius to this fraction of the frame's
    // short side. A lost target can't teleport across the screen, so re-acquiring a
    // far object is almost always wrong. Lower = stricter (fewer far mis-locks,
    // more genuine losses left for a re-click). Raise toward old behavior with care.
    float reacquire_max_frac = 0.2f;

    // How far the best re-acquire candidate's appearance cosine must beat the
    // runner-up's to re-lock. Refuses an ambiguous match when two nearby objects
    // look almost equally like the target. Higher = stricter (won't gamble on a
    // coin-flip; stays LOST for a re-click instead).
    float reacquire_margin = 0.10f;

    // Locked-ROI fast-path (the "DJI" lever): once a target is LOCKED, detect a
    // cropped window around its predicted position instead of running full-frame
    // SAHI every frame — far cheaper, and a tight crop is stretched up so the
    // target gets MORE detail. Full SAHI still runs periodically (and whenever
    // not solidly locked) to catch new objects and recover from drift.
    bool  roi_fastpath       = true;   // enable the locked ROI fast-path
    int   roi_full_interval  = 20;     // run full SAHI every Nth frame while locked
    float roi_scale          = 3.0f;   // ROI side = roi_scale * target max-dimension
};

// Appearance ReID for re-acquiring the locked target.
struct ReidCfg {
    std::string kind       = "histogram";   // histogram (model-free) | onnx (neural)
    std::string model_path = "";            // ReID ONNX (kind: onnx)
    std::string backend    = "onnxruntime"; // onnxruntime | tensorrt (kind: onnx)
    std::string device     = "cuda";        // cuda | cpu
    std::string precision  = "fp16";        // fp32 | fp16 | int8 (tensorrt only)
    int         input_w    = 128;           // ReID model input width  (OSNet: 128)
    int         input_h    = 256;           // ReID model input height (OSNet: 256)
};

// Structured per-frame output of the locked target.
struct OutputCfg {
    std::string sink = "none";         // none | jsonl
    std::string path = "out/track.jsonl";
};

// Top-level runtime configuration. Grows with later milestones.
struct Config {
    DetectorCfg detector;
    TilingCfg   tiling;
    TrackerCfg  tracker;
    LockCfg     lock;
    ReidCfg     reid;
    OutputCfg   output;
    int work_height = 640;              // working resolution (downscale target); 0 = native
    std::string decoder = "auto";       // video decode backend: auto | nvdec | software
};

// Loads a Config from a YAML file. Missing keys fall back to the defaults above.
// Throws std::runtime_error if the file cannot be read or parsed.
Config load_config(const std::string& yaml_path);

}  // namespace ot
