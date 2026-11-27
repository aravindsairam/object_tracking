<h1 align="center">🎯 SkyLock</h1>

<p align="center"><b>Click-to-lock target tracking and ranging for drone video</b></p>

Click a person or vehicle in a drone video and SkyLock stays locked on it. If it
loses the target it says so, instead of drifting onto something else. Real-time
C++ on the GPU.

## Demo

<video src="https://github.com/user-attachments/assets/918c238e-250e-450c-a7ef-56439afaee70" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/8e37e3f4-de5b-4e32-bb85-b4fd31aff939" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/14598d87-b5f9-4de1-a26a-2b1b47fa4dc0" controls muted loop width="800"></video>

**With live distance** from the drone's telemetry:

<video src="https://github.com/user-attachments/assets/70cd8431-bad0-43e9-8f0f-f4f751b7e86a" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/23e1d421-8fcc-48d3-b6f5-6aa8789e4990" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/478876da-7214-4a5b-be90-b66c92254711" controls muted loop width="800"></video>

## Features

- **Click to lock** one target and follow it until you reset.
- **Stays on the right object** by checking every frame that it still looks like what you clicked.
- **Honest status:** `LOCKED`, `COASTING` or `LOST`.
- **Small, distant targets** are found by scanning the frame in tiles.
- **Distance and map position** of the target, from drone telemetry.
- **Swappable models:** RF-DETR, YOLO, WALDO or UniDrone, picked in a YAML file.

## Quick start

**You need:** Linux, a GPU with CUDA 12 (CMake expects `/usr/local/cuda-12.6`),
cuDNN 9, CMake ≥ 3.18 and a C++17 compiler.

```bash
# 1. System packages (Ubuntu) + TensorRT
sudo apt install build-essential cmake libopencv-dev libyaml-cpp-dev \
                 libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
                 libcurl4-openssl-dev
pip install tensorrt

# 2. Third-party code (not committed)
git clone https://github.com/Geekgineer/motcpp.git third_party/motcpp
git -C third_party/motcpp checkout b35c189
git -C third_party/motcpp apply ../../patches/motcpp-local-fixes.patch
#    ONNX Runtime 1.23.2 (Linux x64 GPU) from github.com/microsoft/onnxruntime/releases,
#    unpacked into third_party/onnxruntime/, with the cuDNN 9 libs copied into its lib/

# 3. Models: RF-DETR Medium + ReID
pip install rfdetr onnx onnxruntime torch torchreid
python scripts/export_rfdetr_onnx.py medium
python scripts/make_uint8_onnx.py        # -> models/rfdetr_medium_576_u8.onnx
python scripts/export_reid_onnx.py       # -> models/reid_osnet_x1_0.onnx

# 4. Build and run
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/object_tracking clip.mp4 configs/rfdetr_medium_sahi_ocsort_u8.yaml
```

The first run takes a few minutes to build the TensorRT engine. If TensorRT isn't
found, add `-DTENSORRT_LIB_DIR=/path/to/tensorrt_libs` to `cmake`.

## Usage

```bash
./build/object_tracking <video | rtsp-url | camera-index> [config.yaml] [options]
```

| Keys and mouse | |
|---|---|
| Click | Lock a target |
| `r` · `space` · `s` · `q` | Reset · pause · screenshot · quit |
| On the map: drag · scroll or `+` `-` · double-click or `f` | Pan · zoom · follow again |

| Option | What it does |
|---|---|
| `--zoom` | Magnified side panel of the target |
| `--map` | Map side panel with the drone and the target |
| `--record FILE` | Save the annotated video |
| `--height N` | Working resolution (lower = faster) |
| `--telemetry FILE` | Drone telemetry file (default: found next to the video) |
| `--gimbal-pitch DEG` | Fixed camera pitch (0 = level, -90 = straight down) |
| `--heading-offset DEG` | Correct the camera heading by hand |

Run with no arguments for the full list.

## Configuration

Settings live in YAML presets in `configs/`. Start from
`rfdetr_medium_sahi_ocsort_u8.yaml`. Misspelled keys are rejected.

| Section | Controls |
|---|---|
| `detector` | Model, backend (`onnxruntime` / `tensorrt`), confidence |
| `tiling` | Tile size and overlap |
| `tracker` | `bytetrack`, `botsort` or `ocsort` |
| `lock` | Look-check strictness, time before `LOST` |
| `reid` | Appearance model: `histogram` or `onnx` |
| `range` | Distance estimation |
| `map` | Map tiles, cache folder, offline mode |

## How it works

```
video ─▶ detect ─▶ track ─▶ lock ─▶ overlay
                              │
telemetry ────────────────────┴─▶ distance ─▶ map
```

A matching tracker ID is never trusted alone: the target must keep looking like
what you clicked. After a loss, the lock only returns to a clear best match near
the last position.

## Distance to the target

Keep the drone's telemetry file next to the video with the same name.

| Shown as | Method | Needs |
|---|---|---|
| `~143m` | Camera ray meets flat ground | Camera pitch |
| `~143m(est)` | Typical object size vs size on screen | GPS, altitude, focal length |

`(est)` is about ±10–15 %. Implausible values are hidden. No telemetry? Set
`range.equiv_focal_mm` to use the size method.

## Target on the map

Run with `--map`.

| You see | When |
|---|---|
| Drone, its track, camera view wedge | Always (needs GPS) |
| Target marker and trail | A target is locked |
| Ring around the drone | Locked, but camera direction unknown |

- The position is an estimate: drone GPS + distance + camera direction.
- Camera direction is read from the video file's own metadata. If the camera is
  panned sideways for the whole clip, set `--heading-offset DEG`.
- Map tiles come from OpenStreetMap the first time an area is shown, then from
  `~/.cache/map_tiles`. Set `map.offline: true` to never download.
- Also written to the JSONL: `target_lat`, `target_lon`, `bearing_deg`.

## Accuracy check: TensorRT FP16

The FP16 engine was checked on all 5,000 COCO val2017 images and loses nothing:

| Backend | COCO AP | AP50 |
|---|---|---|
| ONNX Runtime CUDA (reference) | 54.68 | 73.45 |
| TensorRT FP32 | 54.73 | 73.50 |
| **TensorRT FP16 (production)** | **54.74** | **73.57** |

<details>
<summary>Reproduce</summary>

```bash
# Drone footage: compare FP16 against the CUDA reference
./build/parity_dump video clip.MP4 configs/rfdetr_medium_sahi.yaml \
    --out fp16.jsonl --every 15 --max 200 --conf 0.10
./build/parity_dump video clip.MP4 configs/rfdetr_medium_sahi.yaml \
    --out ref.jsonl --every 15 --max 200 --conf 0.10 --backend onnxruntime
python3 tools/parity_compare.py --ref ref.jsonl --test fp16.jsonl \
    --video clip.MP4 --outdir vis/

# COCO val2017 mAP (pip install pycocotools)
./build/parity_dump coco path/to/val2017 configs/rfdetr_medium_sahi.yaml \
    --out coco_fp16.jsonl
python3 tools/coco_eval_dump.py --ann instances_val2017.json --dump coco_fp16.jsonl
```

</details>

## Acknowledgments

Tracking (ByteTrack, BoT-SORT, OC-SORT) comes from
[motcpp](https://github.com/Geekgineer/motcpp).
