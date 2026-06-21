#!/usr/bin/env python
"""Export an OSNet ReID model to ONNX for the C++ object_tracking pipeline.

Produces a feature-extractor ONNX the C++ `OnnxReidEmbedder` runs to re-identify
the locked target. Output is a per-crop embedding (L2-normalized on the C++ side).

Install:
  pip install torch torchreid onnx           # CPU torch is fine for export

Weights:
  --weights is optional. Without it, ImageNet-pretrained OSNet is used (works, but
  weaker for ReID). For real person-ReID accuracy, download a Market1501/MSMT OSNet
  checkpoint from the torchreid model zoo and pass it with --weights.

Usage:
  python scripts/export_reid_onnx.py                      -> models/reid_osnet_x1_0.onnx
  python scripts/export_reid_onnx.py --weights osnet_x1_0_market.pth
  python scripts/export_reid_onnx.py --name osnet_x0_25 --out models/reid_osnet_x0_25.onnx

The exported ONNX matches the C++ OnnxReidEmbedder / ReidCfg defaults:
  input  "input"    : float32 [1,3,256,128]  ImageNet-normalized RGB
  output "features" : float32 [1,D]           appearance embedding (D=512 for x1_0)
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export OSNet ReID to ONNX")
    ap.add_argument("--name", default="osnet_x1_0",
                    help="torchreid model name (e.g. osnet_x1_0, osnet_x0_25)")
    ap.add_argument("--weights", default=None, help="optional ReID .pth checkpoint")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default=None, help="output .onnx path")
    args = ap.parse_args()

    try:
        import torch
        import torchreid
    except ImportError as e:
        missing = getattr(e, "name", None)
        print(f"error: missing module '{missing}' ({e})", file=sys.stderr)
        print("install deps:  pip install torch torchreid onnx gdown", file=sys.stderr)
        if missing and missing not in ("torch", "torchreid", "onnx"):
            print(f"(torchreid's dep list is loose — also: pip install {missing})", file=sys.stderr)
        sys.exit(1)

    model = torchreid.models.build_model(
        name=args.name, num_classes=1000, pretrained=args.weights is None)
    if args.weights:
        torchreid.utils.load_pretrained_weights(model, args.weights)
    model.eval()  # eval mode -> forward() returns the feature embedding, not logits

    out = Path(args.out) if args.out else Path("models") / f"reid_{args.name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    import inspect
    dummy = torch.randn(1, 3, args.height, args.width)
    kwargs = dict(
        input_names=["input"], output_names=["features"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    )
    # torch >=2.9 defaults to the dynamo exporter (needs onnxscript); force the
    # legacy TorchScript path so this works with a plain `pip install torch`.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, str(out), **kwargs)
    print(f"exported: {out}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        print(f"  input  {i.name:9s} {i.shape} {i.type}")
        print(f"  output {o.name:9s} {o.shape} {o.type}")
        print(f"  -> set reid.input_h={args.height} reid.input_w={args.width} in the config")
    except ImportError:
        print("(skip verify: onnxruntime not installed)")


if __name__ == "__main__":
    main()
