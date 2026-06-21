#!/usr/bin/env python
"""Export a YOLO .pt to ONNX with a DYNAMIC batch dimension.

Dynamic batch lets us feed all SAHI tiles through the detector in a single
batched inference call instead of one call per tile (the big speed win).

Usage:
  python export_yolo_onnx.py <model.pt> [out_dir]
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: export_yolo_onnx.py <model.pt> [out_dir]", file=sys.stderr)
        sys.exit(2)
    pt = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else pt.parent

    model = YOLO(str(pt))
    exported = Path(model.export(format="onnx", dynamic=True, simplify=True,
                                 opset=12, imgsz=640))

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (pt.stem + "_dyn.onnx")
    if exported.resolve() != dest.resolve():
        shutil.copy2(exported, dest)
    print("exported:", dest)


if __name__ == "__main__":
    main()
