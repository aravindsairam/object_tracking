<h1 align="center">🎯 SkyLock</h1>

<p align="center"><b>Real-time click-to-lock target tracking &amp; ranging for drone video</b></p>

A C++ application that detects people and vehicles in any video and lets you
**lock a single target by clicking it**, then follows only that target. It
survives occlusion, camera motion, and similar-looking objects nearby, and it
tells you honestly when it has lost the target instead of silently drifting onto
the wrong thing.

It runs in real time on NVIDIA RTX GPUs. Drone telemetry is **optional**: with
no telemetry everything works as-is, and when a DJI `.SRT` sidecar sits next to
the video it additionally estimates the **distance to the locked target**. The
output is an on-screen overlay plus a structured per-frame record on disk.

---

## Demo

A single clicked target stays locked through camera motion and nearby look-alikes,
with the animated reticle marking the locked object:

<video src="https://github.com/user-attachments/assets/918c238e-250e-450c-a7ef-56439afaee70" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/8e37e3f4-de5b-4e32-bb85-b4fd31aff939" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/14598d87-b5f9-4de1-a26a-2b1b47fa4dc0" controls muted loop width="800"></video>

### Tracking with live distance

With a DJI `.SRT` sidecar next to the video, the lock also carries a live
distance estimate — `~Nm` when gimbal attitude is available (ground-plane),
`~Nm(est)` from the class-size prior when it isn't (see
[Distance to the locked target](#distance-to-the-locked-target-dji-telemetry)):

<video src="https://github.com/user-attachments/assets/70cd8431-bad0-43e9-8f0f-f4f751b7e86a" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/23e1d421-8fcc-48d3-b6f5-6aa8789e4990" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/478876da-7214-4a5b-be90-b66c92254711" controls muted loop width="800"></video>

---

## How it works

```
  video / RTSP / camera
          │
   ┌──────▼───────┐   decode (NVDEC or software), downscale to working height
   │ VideoSource  │
   └──────┬───────┘
          │ frame
   ┌──────▼───────┐   find every person/vehicle box (YOLO / WALDO / RF-DETR,
   │  Detector    │   optionally SAHI-tiled for tiny/distant objects)
   └──────┬───────┘
          │ detections
   ┌──────▼───────┐   give each object a stable ID across frames
   │  MotTracker  │   (ByteTrack / BoT-SORT / OC-SORT, config-selectable)
   └──────┬───────┘
          │ tracks
   ┌──────▼───────┐   you click one → it becomes THE target. State machine keeps
   │ LockManager  │   it locked, coasts through brief gaps, re-acquires by
   │  + ReID      │   appearance after occlusion, or honestly reports LOST.
   └──────┬───────┘
          │ TargetState (box, velocity, confidence, ids)
   ┌──────▼───────┐
   │ overlay +    │   draw on screen  +  append one JSON line per frame to disk
   │ JSONL sink   │
   └──────────────┘
```

The four stages run across **three threads** (capture, inference, display), with
the display decoupled from inference so the video always plays smoothly even when
a heavy frame takes longer to process.

**The hard part is staying locked on the right one**, not detecting objects, and
the core idea is that a tracker's own confidence is never trusted. Once you click
a target, the `LockManager` runs a small state machine each frame:

- **LOCKED**: following the real target.
- **COASTING**: briefly missing; the box is predicted from velocity.
- **LOST**: gave up honestly; waiting for re-lock or `r`.

Because a tracker will happily reuse an ID for a *different* object, a surviving
ID is not proof of identity. So the lock is gated by appearance: the matched box
must keep resembling a stored ReID template (an HSV histogram or a neural OSNet
model), and re-acquiring after a loss requires a strong, unambiguous match near
the target's last known spot. Once solidly locked, it detects only a small crop
around the target instead of the full frame, which is much faster.

### Pluggable detectors

Two axes, both set in YAML and swappable without recompiling: the **model
family** (`yolo` / `waldo` / `unidrone` / `rfdetr`) and the **backend**
(`onnxruntime` or `tensorrt`). A fine-tuned model exported to a known family is
a pure config change.

---

## Distance to the locked target (DJI telemetry)

If the video came from a DJI drone, keep the flight's `.SRT` subtitle file next
to the video (auto-detected; or pass `--srt FILE`). A second HUD line then shows
the drone's own GPS / altitude / focal length, and while a target is locked the
reticle, zoom panel, and JSONL also carry an approximate **distance to it**.

A single camera cannot measure range without one extra constraint, so the
estimator runs an if/else per frame and uses the best constraint available:

| Overlay tag | Technique | Needs |
|---|---|---|
| `~143m` | **Ground-plane ray intersection** — cast the pixel ray through the box's bottom-center (the ground contact point), rotate it by the gimbal attitude, intersect flat ground: `slant = altitude / sin(depression)`. No assumption about the object itself. | Gimbal pitch: `--gimbal-pitch DEG` (fixed, 0 = level / -90 = nadir), or per-frame `gb_pitch`/`gb_roll` parsed automatically from enterprise-DJI SRTs (M30/M300/M350). |
| `~143m(est)` | **Class-size prior** — `slant = f_px × typical_size(class) / box_px`, with typical longest dimensions per class (car ≈ 4.6 m, person ≈ 1.7 m, …; editable under `range.size_priors`). | Nothing beyond the SRT. This is the fallback when no attitude exists — consumer Mavic Fly-app SRTs carry **no** gimbal fields. |

Details that make it behave:

- **Zoom-proof intrinsics.** `f_px` comes from the SRT's *per-frame*
  35 mm-equivalent `focal_len`, so optical/digital zoom ramps mid-clip are
  handled; all math runs at the working resolution (`f_px` scales with width).
- **Smoothing & honesty.** The output is a rolling median
  (`range.smooth_window` frames) that resets on a new lock and on a method
  change; near-horizon rays (where `h/tan` blows up), ascending rays, and fixes
  past `range.max_range_m` fall through instead of showing wild numbers, and
  nothing is shown when no method applies. LOST targets get no estimate.
- **Accuracy.** `est` is prior-dominated (~±10–15 %). Ground-plane is limited
  by pitch accuracy and by the barometric `rel_alt` being relative to the
  takeoff point — set `range.ground_alt_offset_m` when the target's ground
  differs from the launch elevation.
- **No SRT at all?** Set `range.equiv_focal_mm` (35 mm-equivalent) in the config
  to enable the size-prior method on any footage.
- JSONL rows gain `range_slant_m`, `range_ground_m`, and
  `range_method` (`"gp"` / `"est"`).

Per-frame attitude from PX4 telemetry (ulog → CSV, live MAVLink/ROS2) is the
planned next source and slots into the same precedence chain.
Verify the math without a GPU: `./build/test_range [flight.SRT]`.

---

## Setup

### Prerequisites

- Linux with an **NVIDIA GPU** + CUDA 12.x (CPU works but is slow).
- **CMake ≥ 3.18**, a C++17 compiler (GCC/Clang).
- **OpenCV 4** (`videoio, highgui, imgproc, dnn, video`) and **yaml-cpp**.
- **FFmpeg** dev libraries (`libavformat libavcodec libavutil libswscale`) for
  the NVDEC hardware-decode path.
- **ONNX Runtime (GPU build)** vendored at `third_party/onnxruntime/`.
- **motcpp** (ByteTrack/BoT-SORT/OC-SORT) vendored at `third_party/motcpp/`
  (upstream commit `b35c189`) with our fixes applied from
  `patches/motcpp-local-fixes.patch` (IoU-variant bugs + buffered IoU):
  `git -C third_party/motcpp apply ../../patches/motcpp-local-fixes.patch`.
  CMake refuses to configure against an unpatched copy.
- For the **TensorRT** backend: the `tensorrt` pip wheel (provides
  `libnvinfer.so.10`). Without it, the TensorRT backend silently falls back to
  the slower CUDA EP.

Example system packages on Ubuntu:

```bash
sudo apt install build-essential cmake libopencv-dev libyaml-cpp-dev \
                 libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
pip install tensorrt
```

### Get the models

Model files live in `models/` and are exported with the helper scripts. Export
only what you plan to run. The RF-DETR scripts produce
`models/rfdetr_small_512.onnx` and `models/rfdetr_medium_576.onnx`; the ReID
script produces `models/reid_osnet_x1_0.onnx`.

```bash
pip install rfdetr onnx onnxruntime
python scripts/export_rfdetr_onnx.py small
python scripts/export_rfdetr_onnx.py medium

pip install torch torchreid onnx
python scripts/export_reid_onnx.py

python scripts/export_yolo_onnx.py
```

> The WALDO and UniDrone ONNX files may already be present in `models/`. The
> first run of a TensorRT config is **slow**: it builds and caches a TensorRT
> engine under `models/trt_engine_cache/`; every run after that is fast.

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/object_tracking` (plus a few headless smoke/test tools). The
binary embeds an RPATH to the vendored ONNX Runtime and CUDA libs, so you do not
need to set `LD_LIBRARY_PATH`. If TensorRT lives somewhere non-standard, pass
`-DTENSORRT_LIB_DIR=/path/to/tensorrt_libs` to the first `cmake`.

---

## Run

```bash
./build/object_tracking <video|rtsp|camera-index> [config.yaml] [options]
```

Examples:

```bash
./build/object_tracking clip.mp4
./build/object_tracking clip.mp4 configs/rfdetr_medium.yaml --zoom --record out.mp4
./build/object_tracking clip.mp4 configs/rfdetr_medium_sahi.yaml --start 500
./build/object_tracking rtsp://cam/stream configs/waldo.yaml --autolock --display-fps 0 --profile
```

If you pass a `.yaml` path it becomes the config; otherwise it defaults to
`configs/waldo.yaml`. The first argument is the video source (a file path, an
RTSP URL, or a camera index like `0`).

Click to lock the target under the cursor; press `r` to reset the lock, `space`
to pause, `s` to screenshot, `q`/`Esc` to quit.

### Command-line options

The common ones (run with no args for the full list):

| Option | Description |
|--------|-------------|
| `--height N` | Working resolution: downscale the frame to N px tall. Lower = faster, higher = more small-object detail. |
| `--start N` | Seek to frame N before starting. |
| `--record FILE` | Write the annotated video to FILE (e.g. `out.mp4`). |
| `--autolock` | Automatically lock the largest/most-central track, for headless runs. |
| `--zoom` | Show a magnified side panel of the locked target. |
| `--srt FILE` | DJI SRT telemetry sidecar (default: auto-detect `<video>.SRT`). |
| `--gimbal-pitch DEG` | Fixed gimbal pitch for ground-plane ranging (0 = level, -90 = nadir); overrides SRT attitude. |

---

## Configuration (YAML)

Every model and behavior is set in YAML so you can swap models and tune the lock
without recompiling. Pick a preset in `configs/` and edit it; missing keys fall
back to sensible defaults. The keys you'll touch most:

- **`detector`**: `family` (`yolo`/`waldo`/`unidrone`/`rfdetr`), `model_path`,
  `backend` (`onnxruntime`/`tensorrt`), `input_size`, `conf` (score threshold),
  `class_map`.
- **`tiling`**: `enabled`, `tile` size, `overlap`, `full_frame`. Cost scales
  with tile count, so fewer tiles = faster.
- **`tracker`**: `type` (`bytetrack`/`botsort`/`ocsort`) plus the shared
  association knobs (`track_thresh`, `match_thresh`, `track_buffer`,
  `iou_threshold`). ByteTrack is fastest; BoT-SORT is steadiest under camera
  motion; OC-SORT is the best motion-only fit for erratic aerial targets.
- **`lock`**: `verify_thresh` (per-frame appearance gate that stops
  ID-switches), `reacquire_thresh`/`reacquire_margin` (re-lock strictness),
  `coast_to_lost` (frames before declaring LOST), `roi_fastpath` /
  `roi_full_interval` (the speed lever).
- **`reid`**: `kind` (`histogram` or neural `onnx` OSNet) and its `model_path`.
- **`input` / `output`**: `input.height` (working resolution), `output.sink`
  (`none`/`jsonl`) and `output.path` for the per-frame JSONL record.
- **`range`**: distance-to-target estimation — `enabled`, `size_priors`
  (typical class sizes in meters), `smooth_window`, `max_range_m`,
  `min_depression_deg`, `ground_alt_offset_m`, `equiv_focal_mm` (for videos
  without an SRT).

---

## Accuracy audit: does the TensorRT FP16 engine lose detections?

Deployed DETR engines can lose accuracy **silently**. Two documented failure
modes motivated this audit: TensorRT has miscompiled deformable-attention
`GridSample` in context (≈ −10 AP on D-FINE, with confidence scores left
intact, so nothing looks wrong on screen), and the global FP16 builder flag can
place lossy casts on precision-critical box math (≈ −7 AP more). Our exported
RF-DETR graph contains both trigger patterns — 4 `GridSample` nodes in the
decoder cross-attention and the weak-typed `trt_fp16_enable` flag — and until
now the shipping engine had only ever been sanity-checked by eye. Score-only
checks are provably blind to this class of bug (boxes drift while scores stay
plausible), so the audit compares **boxes**, not scores.

### What we did

Both tiers run through the real C++ pipeline via `tools/parity_dump.cpp`
(built with the normal build):

1. **Backend parity on our own footage (label-free).** Three 4K drone clips,
   every 15th frame, the exact production path (SAHI tiling + IOS merge), run
   identically through three backends: ORT **CUDA EP** (the non-TensorRT
   reference), **TRT fp32** (isolates TensorRT compilation errors), and
   **TRT fp16** (production). `tools/parity_compare.py` greedy-matches
   detections per frame by class + IoU, separates *true* disagreements from
   near-threshold flips, and renders the worst-disagreement frames as
   annotated PNGs for visual inspection.
2. **Absolute mAP on COCO val2017 (all 5,000 images).** Batch-1 inference with
   RF-DETR's official postprocess (top-K over all query×class scores), scored
   with pycocotools by `tools/coco_eval_dump.py` — directly comparable to the
   published RF-DETR-Medium reference (54.7 AP / 73.6 AP50).

### What we found — the engine is clean

**Tier 1** (≈25,000 matched detections across the three clips, conf 0.3):

| vs CUDA-EP reference | matched IoU (mean) | true misses | true extras | mean box drift |
|---|---|---|---|---|
| TRT fp32 | 0.998–0.999 | 0.04–0.08% | 0.03–0.08% | 0.01 px |
| TRT fp16 (production) | 0.977–0.992 | 0.12–0.31% | 0.09–0.31% | 0.11 px |

fp32-TRT ≈ CUDA-EP rules out a TensorRT miscompile of this graph; fp16 adds
only textbook numerical noise. Rendering the worst frames confirms the
remaining disagreements are tiny distant objects flickering around the 0.3
confidence cut — no drift, no hallucinations, no class swaps.

**Tier 2** (COCO val2017, full 5,000 images):

| backend | AP | AP50 | AP75 | AP-small |
|---|---|---|---|---|
| ORT CUDA EP | 54.68 | 73.45 | 59.05 | 35.94 |
| TRT fp32 | 54.73 | 73.50 | 59.04 | 35.93 |
| **TRT fp16 (production)** | **54.74** | **73.57** | 59.00 | 35.82 |

All three backends agree within 0.06 AP — and they land exactly on the
official RF-DETR-Medium reference (54.7 AP / 73.6 AP50), certifying the whole
chain (ONNX export → preprocessing → engine → decode) end to end.

**Takeaway:** TRT-FP16 is accuracy-neutral for this export; the feared silent
losses do not apply. These tools are the standing regression gate — rerun them
after any model re-export, EP flag change, or TensorRT upgrade.

### Reproduce

```bash
# Tier 1: per-backend dumps over a clip (video mode = the exact production path)
./build/parity_dump video clip.MP4 configs/rfdetr_medium_sahi.yaml \
    --out fp16.jsonl --every 15 --max 200 --conf 0.10
./build/parity_dump video clip.MP4 configs/rfdetr_medium_sahi.yaml \
    --out ref.jsonl  --every 15 --max 200 --conf 0.10 --backend onnxruntime
python3 tools/parity_compare.py --ref ref.jsonl --test fp16.jsonl \
    --video clip.MP4 --outdir vis/          # stats + worst-frame PNG gallery

# Tier 2: COCO val2017 mAP (pip install pycocotools)
./build/parity_dump coco path/to/val2017 configs/rfdetr_medium_sahi.yaml \
    --out coco_fp16.jsonl
python3 tools/coco_eval_dump.py --ann instances_val2017.json --dump coco_fp16.jsonl
```

---

## Acknowledgments

Multi-object tracking (ByteTrack, BoT-SORT, OC-SORT) is provided by the vendored
[motcpp](https://github.com/Geekgineer/motcpp) library.
