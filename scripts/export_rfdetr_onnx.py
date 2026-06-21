#!/usr/bin/env python
"""Export an RF-DETR variant to ONNX for the C++ object_tracking pipeline.

RF-DETR ships only as a PyTorch package; the C++ runtime needs ONNX. This script
loads a variant (weights auto-download from HuggingFace on first run), exports a
single FP32 ONNX, and copies it into ./models with a stable name. Precision
(FP16 / INT8) is chosen later at TensorRT-engine-build time in C++, so one FP32
ONNX serves all precisions.

Install (CPU is fine for export):
  pip install rfdetr onnx onnxruntime

Usage:
  python scripts/export_rfdetr_onnx.py small               -> models/rfdetr_small_512.onnx
  python scripts/export_rfdetr_onnx.py medium              -> models/rfdetr_medium_576.onnx
  python scripts/export_rfdetr_onnx.py medium --checkpoint /path/to.pth
  python scripts/export_rfdetr_onnx.py small --out-dir models --resolution 512

The exported ONNX matches what src/detect/rfdetr_detector.cpp decodes:
  input  "input"  : float32 [1,3,H,W]  ImageNet-normalized RGB, stretch-resized
  output "boxes"  : float32 [1,Q,4]    cxcywh, normalized to [0,1]
  output "labels" : float32 [1,Q,C]    raw class logits (apply sigmoid)
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# variant -> (rfdetr class name, native square resolution)
VARIANTS = {
    "small":  ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RF-DETR to ONNX")
    ap.add_argument("variant", choices=sorted(VARIANTS), help="RF-DETR variant")
    ap.add_argument("--checkpoint", default=None,
                    help="optional .pth weights (default: package auto-downloads)")
    ap.add_argument("--out-dir", default="models", help="destination dir for the .onnx")
    ap.add_argument("--resolution", type=int, default=None,
                    help="override square input size (must be valid for the variant)")
    ap.add_argument("--no-dynamic-batch", dest="dynamic_batch", action="store_false",
                    help="bake a fixed batch=1 (default: dynamic batch, so SAHI tiles "
                         "run in one batched inference)")
    ap.set_defaults(dynamic_batch=True)
    args = ap.parse_args()

    class_name, default_res = VARIANTS[args.variant]
    res = args.resolution or default_res

    try:
        import rfdetr  # noqa: F401
        model_cls = getattr(__import__("rfdetr", fromlist=[class_name]), class_name)
    except ImportError as e:
        print(f"error: {e}\ninstall with: pip install rfdetr onnx onnxruntime", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if args.checkpoint:
        kwargs["pretrain_weights"] = args.checkpoint
    print(f"Loading {class_name} (resolution {res}) ...")
    model = model_cls(**kwargs)

    # Export into a temp dir, then copy to a stable name. The package writes
    # `inference_model.onnx` (and sometimes a sibling .sim.onnx).
    tmp = Path(tempfile.mkdtemp(prefix="rfdetr_export_"))
    print(f"Exporting to ONNX (shape={res}x{res}, dynamic_batch={args.dynamic_batch}) ...")
    model.export(output_dir=str(tmp), shape=(res, res), dynamic_batch=args.dynamic_batch)

    produced = sorted(tmp.rglob("*.onnx"), key=lambda p: len(p.name))  # prefer base name
    if not produced:
        print(f"error: no .onnx produced under {tmp}", file=sys.stderr)
        sys.exit(1)
    src = produced[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"rfdetr_{args.variant}_{res}.onnx"
    shutil.copy2(src, dest)
    print(f"exported: {dest}")

    _verify(dest)


def _verify(onnx_path: Path) -> None:
    """Print the ONNX I/O contract and the COCO class indexing the C++ decoder
    relies on. RF-DETR's raw ONNX emits labels in the 91-slot COCO category-id
    space (1-indexed, id 0 = background), so the C++ side uses the `coco91`
    class_map (keep ids {1,2,3,4,6,8}) — NOT the contiguous 80-class `coco` map."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("(skip verify: onnxruntime not installed)")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("\nONNX I/O contract:")
    for i in sess.get_inputs():
        print(f"  input  {i.name:8s} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name:8s} {o.shape} {o.type}")
    n_cls = next((o.shape[-1] for o in sess.get_outputs() if o.shape[-1] != 4), None)
    print(f"  -> class slots: {n_cls} (expect 91 for the coco91 class_map)")

    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
        get = (lambda i: COCO_CLASSES.get(i)) if isinstance(COCO_CLASSES, dict) \
            else (lambda i: COCO_CLASSES[i] if i < len(COCO_CLASSES) else None)
        print("  class ids 1..8:", {i: get(i) for i in range(1, 9)})
        if get(1) == "person":
            print("OK: id 1 == 'person' — matches the C++ `coco91` map "
                  "(keep {1,2,3,4,6,8} = person/bicycle/car/motorcycle/bus/truck).")
        else:
            print("WARNING: id 1 is not 'person' — adjust the coco91 keep ids "
                  "in src/detect/class_map.cpp to match this model.")
    except Exception as e:  # noqa: BLE001
        print(f"(skip class-index check: {e})")


if __name__ == "__main__":
    main()
