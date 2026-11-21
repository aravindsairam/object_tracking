<h1 align="center">🎯 SkyLock</h1>

<p align="center"><b>Click-to-lock target tracking and ranging for drone video</b></p>

Click a person or vehicle in a drone video and SkyLock stays locked on it through
camera motion, short occlusions and look-alikes nearby. If it truly loses the
target it says so, instead of drifting onto something else. With DJI telemetry it
also estimates the distance to the target. Real-time C++ on NVIDIA GPUs.

## Demo

<video src="https://github.com/user-attachments/assets/918c238e-250e-450c-a7ef-56439afaee70" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/8e37e3f4-de5b-4e32-bb85-b4fd31aff939" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/14598d87-b5f9-4de1-a26a-2b1b47fa4dc0" controls muted loop width="800"></video>

**With live distance** from the DJI `.SRT` file next to the video:

<video src="https://github.com/user-attachments/assets/70cd8431-bad0-43e9-8f0f-f4f751b7e86a" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/23e1d421-8fcc-48d3-b6f5-6aa8789e4990" controls muted loop width="800"></video>

<video src="https://github.com/user-attachments/assets/478876da-7214-4a5b-be90-b66c92254711" controls muted loop width="800"></video>

## Features

- **Click to lock.** One target, followed until you reset.
- **Stays on the right object.** Every frame the lock checks that the target still
  *looks* like what you clicked, so a tracker ID swap can't quietly move it to a
  neighbour.
- **Honest status.** `LOCKED`, `COASTING` (briefly unseen, position predicted) or
  `LOST` (waits for you).
- **Small, distant targets.** The frame is split into tiles so the detector sees
  tiny objects at full resolution.
- **Fast once locked.** Only a small crop around the target is searched; the full
  frame is re-checked every 20 frames.
- **Distance to the target** from DJI telemetry ([details](#distance-to-the-target)).
- **Swappable models.** RF-DETR, YOLO, WALDO or UniDrone, on ONNX Runtime or
  TensorRT, chosen in a YAML file.
- **Per-frame record.** Optional JSON line per frame: box, velocity, state, distance.

## Quick start

**You need:** Linux, an NVIDIA GPU, CUDA 12 (CMake expects `/usr/local/cuda-12.6`),
cuDNN 9, CMake ≥ 3.18 and a C++17 compiler.

```bash
# 1. System packages (Ubuntu) + TensorRT
sudo apt install build-essential cmake libopencv-dev libyaml-cpp-dev \
                 libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
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

The first TensorRT run takes a few minutes to build and cache the engine; later
runs start fast. If TensorRT isn't found, add `-DTENSORRT_LIB_DIR=/path/to/tensorrt_libs`
to the first `cmake`.

## Usage

```bash
./build/object_tracking <video | rtsp-url | camera-index> [config.yaml] [options]
```

**Mouse and keys:** click a target to lock · `r` reset · `space` pause ·
`s` screenshot · `q` / `Esc` quit

| Option | What it does |
|---|---|
| `--zoom` | Magnified side panel of the locked target |
| `--record FILE` | Save the annotated video |
| `--start N` | Start at frame N |
| `--height N` | Working resolution (lower = faster, higher = more small-object detail) |
| `--profile` | Print timing every 120 frames |
| `--srt FILE` | DJI telemetry file (default: `<video>.SRT` next to the video) |
| `--gimbal-pitch DEG` | Fixed camera pitch for distance (0 = level, -90 = straight down) |

Run with no arguments for the full list.

## Configuration

Everything is set in YAML, so models and behaviour change without recompiling.
Start from a preset in `configs/`; `rfdetr_medium_sahi_ocsort_u8.yaml` is the
recommended one. Unknown or misspelled keys are rejected with an error.

| Section | Controls |
|---|---|
| `detector` | Model family, model file, backend (`onnxruntime` / `tensorrt`), confidence threshold |
| `tiling` | Tile size and overlap (fewer tiles = faster) |
| `tracker` | `bytetrack` (fastest), `botsort` (steadiest with camera motion), `ocsort` (best for erratic aerial targets) |
| `lock` | How strict the look-check is, how long to coast before `LOST`, the locked fast path |
| `reid` | Appearance model: colour `histogram` or neural `onnx` (OSNet) |
| `range` | Distance estimation: typical object sizes, smoothing, limits |
| `input` / `output` | Working resolution; JSONL output path |

## How it works

```
video ─▶ decode ─▶ detect objects ─▶ track IDs ─▶ lock + appearance check ─▶ overlay + JSONL
         (NVDEC)   (tiled, TensorRT)  (OC-SORT)    (LOCKED / COASTING / LOST)
```

Capture, inference and display run on separate threads, so playback stays smooth
even when a frame takes longer to process.

The hard part is staying on the *right* object. Trackers sometimes hand an ID to
a different object, so a matching ID is never trusted on its own: the target must
keep resembling the appearance template stored when you clicked. After a loss,
the lock only re-attaches to a candidate that is near the last known position and
clearly the best match.

## Distance to the target

Keep the DJI `.SRT` subtitle file next to the video (or pass `--srt`). The overlay
then shows the drone's GPS, altitude and focal length, plus a distance to the locked
target:

| Shown as | Method | Needs |
|---|---|---|
| `~143m` | Ray from the camera to the target's ground contact point, intersected with flat ground | Camera pitch: from enterprise DJI SRTs (M30/M300/M350) or `--gimbal-pitch` |
| `~143m(est)` | Known typical size of the class (car ≈ 4.6 m, person ≈ 1.7 m) vs its size on screen | Only the SRT (works with consumer DJI drones) |

<details>
<summary>Accuracy and limits</summary>

- Zoom changes mid-clip are handled: the focal length is read per frame from the SRT.
- Values are median-smoothed; implausible results (near the horizon, beyond
  `range.max_range_m`) are hidden rather than shown wrong. `LOST` targets get none.
- `(est)` is roughly ±10–15 %. Ground-plane accuracy depends on the pitch value and
  on altitude being relative to the take-off point; set `range.ground_alt_offset_m`
  if the target's ground is higher or lower than the take-off spot.
- No SRT at all? Set `range.equiv_focal_mm` to use the size method on any footage.
- Check the maths without a GPU: `./build/test_range [flight.SRT]`.

</details>

## Accuracy check: TensorRT FP16

TensorRT FP16 engines can lose detections silently, so the production engine was
checked against the reference backend on drone footage and on all 5,000 COCO
val2017 images. It loses nothing measurable:

| Backend | COCO AP | AP50 |
|---|---|---|
| ONNX Runtime CUDA (reference) | 54.68 | 73.45 |
| TensorRT FP32 | 54.73 | 73.50 |
| **TensorRT FP16 (production)** | **54.74** | **73.57** |

All three match the published RF-DETR Medium result (54.7 / 73.6). On drone
footage, FP16 boxes match the reference to within 0.11 px on average.

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

Rerun after re-exporting a model or upgrading TensorRT.

</details>

## Acknowledgments

Multi-object tracking (ByteTrack, BoT-SORT, OC-SORT) comes from
[motcpp](https://github.com/Geekgineer/motcpp).
