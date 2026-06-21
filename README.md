# Object Tracking

A C++ application that detects people and vehicles in any video and lets you
**lock a single target by clicking it** — then follows only that target, the way
DJI's ActiveTrack does. It survives occlusion, camera motion, and similar-looking
objects nearby, and it tells you honestly when it has lost the target instead of
silently drifting onto the wrong thing.

It runs in real time on an RTX 3090 and is designed to move to a Jetson later. It
has **no dependency on drone telemetry** — DJI clips are just test footage. The
output is an on-screen overlay plus a structured per-frame record on disk.

---

## What it does (in one picture)

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
   ┌──────▼───────┐   give each object a stable ID across frames (ByteTrack + GMC)
   │  MotTracker  │
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

The pipeline runs as **three threads** — capture (decode), inference
(detect + track + lock), and display/UI. Crucially, the **display is decoupled
from inference**: every decoded frame is shown at a steady video cadence (the
source fps, capped at 60) with the *most recent* detection painted on top, while
inference consumes only the newest frame and drops stale ones. So playback stays
smooth even though a frame's detection cost swings from ~5 ms (locked ROI) to
~20 ms (full SAHI) — the video never waits for inference, and a slow frame only
leaves the overlay one tick stale, which the lock's velocity prediction hides.

---

## How the tracking actually works (the technique)

The hard part isn't detecting objects — it's *staying locked on the right one*.
This is the core idea: **a tracker's own confidence is never trusted.** Locking is
a system-level decision made by the `LockManager`, gated by independent checks.

1. **Detection** — every frame is run through a detector that returns boxes for
   people and vehicles. Detector models are pluggable (see below). For small or
   distant objects, an optional **SAHI** step slices the frame into overlapping
   tiles, detects each tile, and merges the results with NMS — a tiny far-away
   object becomes large enough to detect once it's blown up to the network input.

2. **Multi-object tracking (MOT)** — all detections go through **ByteTrack**
   (from the vendored `motcpp`), which assigns a persistent **track ID** to each
   object across frames and uses low-confidence detections to keep a flickering
   object alive. Camera/drone motion is cancelled with **global motion
   compensation (GMC)**, so IDs stay stable even when the whole frame moves.

3. **Locking one target** — you click an object; the lock snaps to the nearest
   track, stores its **appearance template** (a ReID embedding), and enters the
   `Locked` state. From then on a small state machine runs every frame:

   ```
   ACQUIRING ──click──▶ LOCKED ◀────── re-acquired by appearance ──────┐
                          │ target missing / appearance mismatch        │
                          ▼                                             │
                       COASTING  (Kalman-style velocity prediction) ────┘
                          │ missing longer than coast_to_lost frames
                          ▼
                        LOST  (honest: confidence 0, waiting for re-lock or 'r')
   ```

   | State | Box shown | Confidence | Meaning |
   |-------|-----------|-----------|---------|
   | `LOCKED`   | measured | 1.0 | following the real target |
   | `COASTING` | predicted from velocity | decays toward 0 | briefly lost, guessing |
   | `LOST`     | none | 0 | gave up honestly — re-lock or press `r` |

4. **The honesty guards** — ByteTrack will happily reuse a track ID for a
   *different* object that wanders into the predicted spot. So a surviving ID is
   **not** proof of identity. Several cheap referees keep the lock honest:
   - **Appearance verify gate** (`verify_thresh`) — every frame, the matched
     box's appearance must still resemble the stored template. If not, the frame
     is treated as a miss (coast) instead of silently following an ID-switch.
   - **Re-acquire threshold + margin** (`reacquire_thresh`, `reacquire_margin`) —
     to re-lock after a loss, a candidate must match the template strongly *and*
     clearly beat the runner-up, so two similar nearby objects don't cause a
     coin-flip mis-lock.
   - **Search-radius cap** (`reacquire_max_frac`) — a lost target can't teleport
     across the screen, so re-acquisition only looks near the last known spot.

5. **ReID (re-identification)** — the appearance template above comes from a ReID
   embedder. Two kinds: a model-free **HSV histogram** (no download, weaker), or a
   neural **OSNet** ONNX (stronger, recommended). This is what lets the target be
   re-acquired after it passes behind a tree or building.

6. **Locked-ROI fast-path** (the speed lever) — once solidly locked, instead of
   running full-frame SAHI every frame, the detector runs only on a small crop
   predicted around the target — far cheaper, and the tight crop is stretched up
   so the target actually gains detail. Full SAHI still runs every Nth frame to
   catch new objects and correct drift.

### Why pluggable detectors

Two independent axes, both set in YAML — swap either without recompiling:

- **Model family** (how the network output is decoded): `yolo` / `waldo` /
  `unidrone` (anchor-free YOLO + NMS) and `rfdetr` (transformer, NMS-free).
- **Backend** (the runtime): `onnxruntime` (CUDA or CPU) or `tensorrt` (fastest,
  FP16/INT8). A **class map** normalizes each model's native class IDs (COCO 80,
  COCO 91, WALDO, UniDrone) into the same person/vehicle categories downstream.

A fine-tuned model exported to a known family is a pure config change. Only a
brand-new architecture needs new decode code.

---

## Setup

### Prerequisites

- Linux with an **NVIDIA GPU** + CUDA 12.x (CPU works but is slow).
- **CMake ≥ 3.18**, a C++17 compiler (GCC/Clang).
- **OpenCV 4** (`videoio, highgui, imgproc, dnn, video`) and **yaml-cpp**.
- **FFmpeg** dev libraries (`libavformat libavcodec libavutil libswscale`) — used
  for the NVDEC hardware-decode path.
- **ONNX Runtime (GPU build)** vendored at `third_party/onnxruntime/`.
- **motcpp** (ByteTrack/BoT-SORT) vendored at `third_party/motcpp/`.
- For the **TensorRT** backend: the `tensorrt` pip wheel (provides
  `libnvinfer.so.10`). Without it, the TensorRT backend silently falls back to
  the slower CUDA EP.

Example system packages on Ubuntu:

```bash
sudo apt install build-essential cmake libopencv-dev libyaml-cpp-dev \
                 libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
pip install tensorrt          # only if you want the TensorRT backend
```

### Get the models

Model files live in `models/` and are exported with the helper scripts. Export
only what you plan to run:

```bash
# RF-DETR detectors (transformer, best small-object recall)
pip install rfdetr onnx onnxruntime
python scripts/export_rfdetr_onnx.py small      # -> models/rfdetr_small_512.onnx
python scripts/export_rfdetr_onnx.py medium     # -> models/rfdetr_medium_576.onnx

# OSNet ReID model (needed for kind: onnx re-acquisition)
pip install torch torchreid onnx
python scripts/export_reid_onnx.py              # -> models/reid_osnet_x1_0.onnx

# YOLO / WALDO detectors
python scripts/export_yolo_onnx.py
```

> The WALDO and UniDrone ONNX files may already be present in `models/`. The
> first run of a TensorRT config is **slow** — it builds and caches a TensorRT
> engine under `models/trt_engine_cache/`; every run after that is fast.

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# If TensorRT lives somewhere non-standard:
cmake -S . -B build -DTENSORRT_LIB_DIR=/path/to/tensorrt_libs
```

This produces `build/object_tracking` (plus a few headless smoke/test tools). The
binary embeds an RPATH to the vendored ONNX Runtime and CUDA libs, so you do not
need to set `LD_LIBRARY_PATH`.

---

## Run

```bash
./build/object_tracking <video|rtsp|camera-index> [config.yaml] [options]
```

Examples:

```bash
# WALDO aerial detector (the default config) on a clip
./build/object_tracking clip.mp4

# RF-DETR medium, recording the annotated output, with the zoom panel
./build/object_tracking clip.mp4 configs/rfdetr_medium.yaml --zoom --record out.mp4

# Max small-object recall (SAHI tiling), seeking past the intro
./build/object_tracking clip.mp4 configs/rfdetr_medium_sahi.yaml --start 500

# Live RTSP, headless auto-lock on the most central object, benchmarking
./build/object_tracking rtsp://cam/stream configs/waldo.yaml --autolock --display-fps 0 --profile
```

If you pass a `.yaml` path it becomes the config; otherwise it defaults to
`configs/waldo.yaml`. The first argument is the video source (a file path, an
RTSP URL, or a camera index like `0`).

### Keys (while the window is focused)

| Key | Action |
|-----|--------|
| **click** | lock the target under the cursor |
| `r` | reset the lock (back to ACQUIRING) |
| `space` | pause / resume |
| `s` | save a screenshot (`shot_<frame>.png`) |
| `q` / `Esc` | quit |

### Command-line options

| Option | Description |
|--------|-------------|
| `--height N` | Working resolution: downscale the frame to N px tall (`0` = native). Overrides the config's `input.height`. Lower = faster, higher = more small-object detail. |
| `--start N` | Seek to frame N before starting. |
| `--record FILE` | Write the annotated video to FILE (e.g. `out.mp4`). |
| `--loop` | Restart the stream when it ends. |
| `--profile` | Print pipeline timing (display fps / capture / infer / detection fps / draw / show ms) to stderr every 120 frames. |
| `--autolock` | Automatically lock the largest/most-central track — for centered tracking or headless runs. |
| `--display-fps N` | Pace the **video** to N fps (default: source fps, capped at 60; `0` = uncapped, for benchmarking). The display is decoupled from inference, so this sets the video cadence — the overlay updates independently at whatever rate detection manages. |
| `--zoom` | Show a magnified side panel of the locked target. |
| `--zoom-width N` | Width of the zoom side panel in px (default 360). |

---

## Configuration reference (YAML)

Every model and behavior is set in YAML so you can swap models and tune the lock
without recompiling. Pick one of the presets in `configs/` and edit it. Missing
keys fall back to sensible defaults.

### `detector` — which model and how to run it

| Key | Description |
|-----|-------------|
| `family` | Decode logic: `yolo` \| `waldo` \| `unidrone` \| `rfdetr`. |
| `model_path` | Path to the `.onnx` file. |
| `backend` | Runtime: `onnxruntime` (CUDA/CPU) \| `tensorrt` (fastest). |
| `device` | `cuda` \| `cpu`. |
| `precision` | `fp32` \| `fp16` \| `int8` (TensorRT only; `int8` also needs `int8_calib`). |
| `input_size` | Square network input size (e.g. 512, 576, 640). |
| `conf` | Detection score threshold. Lower keeps flickering targets alive (fewer false LOSTs); higher is stricter. |
| `nms_iou` | NMS IoU for raw-head models (YOLO/WALDO) and cross-tile merge. RF-DETR is NMS-free. |
| `class_map` | Maps native class IDs to person/vehicle: `coco` \| `coco91` \| `waldo` \| `unidrone`. |

### `tiling` — SAHI tiled inference (small/distant objects)

| Key | Description |
|-----|-------------|
| `enabled` | Turn SAHI on/off. |
| `tile` | Tile size in px — match `input_size` so each tile is fed ~1:1 (max detail). |
| `overlap` | Fractional overlap between tiles (e.g. 0.1–0.2). More overlap = more robust seams but more tiles (slower). |
| `full_frame` | Also run one whole-frame pass to catch large/near objects. |

> Cost scales with **tile count**, driven by `input.height`, `overlap`, and
> `full_frame`. Fewer tiles = faster. For max recall: `height 1440`,
> `overlap 0.2`, `full_frame true`. For speed, lower the height and overlap.

### `lock` — single-target lock behavior

| Key | Description |
|-----|-------------|
| `coast_to_lost` | Frames the target may vanish before the lock is declared LOST. |
| `reacquire_thresh` | Appearance cosine similarity needed to auto re-lock after a loss. Higher = refuse weak/wrong re-acquires. |
| `reacquire_margin` | How far the best candidate must beat the runner-up to re-lock — refuses ambiguous matches between two similar objects. |
| `verify_thresh` | Per-frame appearance gate on the matched track. Below it, the frame is a miss — stops the lock silently following a ByteTrack ID-switch onto a different object. `0` disables. |
| `reacquire_max_frac` | Re-acquire search radius = this × the frame's short side. Stops re-locking onto a far object after a loss. Lower = stricter. |
| `roi_fastpath` | Once locked, detect a crop around the target instead of full-frame SAHI every frame (much faster). `false` = always full SAHI. |
| `roi_full_interval` | Run full SAHI every Nth frame while locked (to catch new objects / correct drift). |
| `roi_scale` | ROI side length = this × the target's largest dimension. |

### `reid` — appearance model for re-acquisition

| Key | Description |
|-----|-------------|
| `kind` | `histogram` (model-free HSV, no download) \| `onnx` (neural OSNet, stronger). |
| `model_path` | The ReID ONNX (for `kind: onnx`). |
| `backend` | `onnxruntime` \| `tensorrt`. |
| `device` | `cuda` \| `cpu`. |
| `precision` | `fp32` \| `fp16` \| `int8` (TensorRT only). |
| `input_w` / `input_h` | ReID input size (OSNet: 128 × 256). |

### `input` and `output`

| Key | Description |
|-----|-------------|
| `input.height` | Working resolution (frame is downscaled to this height). CLI `--height` overrides it. |
| `input.decoder` | Video decode backend: `auto` (NVDEC with software fallback) \| `nvdec` \| `software`. The `OT_DECODER` env var overrides this. |
| `output.sink` | `none` \| `jsonl`. |
| `output.path` | Where to write the JSONL record. |

### Included presets

| Config | Use it for |
|--------|-----------|
| `configs/waldo.yaml` | **Default.** WALDO30 aerial detector — best general drone footage. |
| `configs/waldo_sahi.yaml` | WALDO + SAHI at 1440p for small/far objects. |
| `configs/unidrone.yaml` | UniDrone aerial model (people/vehicles/boats). |
| `configs/rfdetr_small.yaml` | RF-DETR small (512px) — fast transformer detector. |
| `configs/rfdetr_medium.yaml` | RF-DETR medium (576px) — strong on small objects (~68 fps). |
| `configs/rfdetr_medium_sahi.yaml` | RF-DETR medium + SAHI — max recall on the tiniest/most distant objects. |
| `configs/yolo_coco.yaml` | Generic YOLO26 COCO model. |

---

## Output

When `output.sink: jsonl`, every frame with an active lock appends one JSON line
to `output.path` (default `out/track.jsonl`):

```json
{"frame":88,"t":1.468,"state":"locked","conf":1.0,
 "box":[926.9,504.1,36.0,58.8],
 "center_px":[944.9,533.5],"center_norm":[0.492,0.494],
 "vel_px_s":[0.0,0.0],"class_id":1,"track_id":4,"lock_id":1}
```

| Field | Meaning |
|-------|---------|
| `frame` / `t` | Frame index and timestamp in seconds. |
| `state` | `locked` \| `coasting` \| `lost`. |
| `conf` | System-level lock confidence in `[0,1]` (decays while coasting, 0 when lost). |
| `box` | `[x, y, w, h]` in working-frame pixels. |
| `center_px` / `center_norm` | Target center in pixels and normalized `[0,1]`. |
| `vel_px_s` | Target velocity in pixels/second. |
| `class_id` | Stabilized (voted) class of the target. |
| `track_id` | Raw MOT track ID (can churn across re-acquires). |
| `lock_id` | Stable ID for this lock, kept across re-acquires. |

This record is what a downstream control loop (gimbal, follow logic) would
consume — but that loop is out of scope here; this app is detection + tracking
only.
