# Object Tracking

A C++ application that detects people and vehicles in any video and lets you
**lock a single target by clicking it**, then follows only that target. It
survives occlusion, camera motion, and similar-looking objects nearby, and it
tells you honestly when it has lost the target instead of silently drifting onto
the wrong thing.

It runs in real time on NVIDIA RTX GPUs and is designed to move to a Jetson
later. It has **no dependency on drone telemetry**. The output is an on-screen
overlay plus a structured per-frame record on disk.

---

## Demo

A single clicked target stays locked through camera motion and nearby look-alikes,
with the animated reticle marking the locked object:

<video src="https://github.com/aravindsairam/object_tracking/raw/main/videos/d1.mp4" controls muted loop width="800"></video>

<video src="https://github.com/aravindsairam/object_tracking/raw/main/videos/d2.mp4" controls muted loop width="800"></video>

<video src="https://github.com/aravindsairam/object_tracking/raw/main/videos/d3.mp4" controls muted loop width="800"></video>

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

## Setup

### Prerequisites

- Linux with an **NVIDIA GPU** + CUDA 12.x (CPU works but is slow).
- **CMake ≥ 3.18**, a C++17 compiler (GCC/Clang).
- **OpenCV 4** (`videoio, highgui, imgproc, dnn, video`) and **yaml-cpp**.
- **FFmpeg** dev libraries (`libavformat libavcodec libavutil libswscale`) for
  the NVDEC hardware-decode path.
- **ONNX Runtime (GPU build)** vendored at `third_party/onnxruntime/`.
- **motcpp** (ByteTrack/BoT-SORT) vendored at `third_party/motcpp/`.
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

---

## Acknowledgments

Multi-object tracking (ByteTrack, BoT-SORT, OC-SORT) is provided by the vendored
[motcpp](https://github.com/Geekgineer/motcpp) library.
