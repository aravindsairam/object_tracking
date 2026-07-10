#!/usr/bin/env python
"""Fold RF-DETR preprocessing into the ONNX graph (the uint8 fast path).

Takes an already-exported FP32 ONNX (scripts/export_rfdetr_onnx.py) and
prepends a preprocessing head, leaving every original node and weight
byte-identical:

    uint8 NHWC BGR  ->  Cast fp32  ->  Transpose NCHW  ->  Gather [2,1,0]
    (BGR->RGB)  ->  x * 1/(255*std)  +  (-mean/std)  ->  original graph

The input contract changes from ImageNet-normalized fp32 NCHW RGB to RAW
RESIZED PIXELS — so the C++ side skips cvtColor + the fp32 normalize entirely
and uploads 4x fewer bytes; TensorRT fuses the head into the first layers.
The input tensor keeps the name "input" (the C++ TrtProfile contract).

    python scripts/make_uint8_onnx.py [models/rfdetr_medium_576.onnx]
        -> models/rfdetr_medium_576_u8.onnx

Ends with an equivalence check: both models run on the same random image
(CPU EP), outputs must match to fp32 tolerance.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "models/rfdetr_medium_576.onnx")
    dest = src.with_name(src.stem + "_u8.onnx")
    model = onnx.load(str(src))
    g = model.graph

    assert g.input[0].name == "input", f"expected input 'input', got {g.input[0].name}"
    dims = g.input[0].type.tensor_type.shape.dim
    batch = dims[0].dim_param or dims[0].dim_value  # keep dynamic batch as-is
    assert dims[1].dim_value == 3, "expected NCHW fp32 input"
    H, W = dims[2].dim_value, dims[3].dim_value
    assert H > 0 and W > 0, "expected static spatial dims"

    # Original consumers of "input" now read the preprocess chain's output.
    for node in g.node:
        node.input[:] = ["preproc.norm" if i == "input" else i for i in node.input]

    del g.input[0]
    g.input.insert(0, helper.make_tensor_value_info(
        "input", TensorProto.UINT8, [batch, H, W, 3]))

    scale = (1.0 / (255.0 * STD)).reshape(1, 3, 1, 1)
    bias = (-MEAN / STD).reshape(1, 3, 1, 1)
    g.initializer.extend([
        numpy_helper.from_array(np.array([2, 1, 0], np.int64), "preproc.bgr2rgb_idx"),
        numpy_helper.from_array(scale.astype(np.float32), "preproc.scale"),
        numpy_helper.from_array(bias.astype(np.float32), "preproc.bias"),
    ])
    head = [
        helper.make_node("Cast", ["input"], ["preproc.f32"],
                         name="preproc/cast", to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["preproc.f32"], ["preproc.nchw"],
                         name="preproc/nchw", perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["preproc.nchw", "preproc.bgr2rgb_idx"],
                         ["preproc.rgb"], name="preproc/bgr2rgb", axis=1),
        helper.make_node("Mul", ["preproc.rgb", "preproc.scale"], ["preproc.scaled"],
                         name="preproc/scale"),
        helper.make_node("Add", ["preproc.scaled", "preproc.bias"], ["preproc.norm"],
                         name="preproc/shift"),
    ]
    for n in reversed(head):
        g.node.insert(0, n)

    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    print(f"wrote {dest}")

    # ---- equivalence check: u8 model(raw) == fp32 model(CPU-normalized) ----
    import onnxruntime as ort
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (1, H, W, 3), dtype=np.uint8)          # NHWC BGR
    rgb = img[0, :, :, ::-1].astype(np.float32)                       # HWC RGB
    # FOLDED normalize x*(1/(255*std)) + (-mean/std) — the exact arithmetic of
    # both the grafted head and the C++ convertTo path (the two-step
    # (x/255-mean)/std differs in the last fp32 bit and the network amplifies
    # that to ~1e-4 at the outputs; against the folded form the u8 model is
    # bit-exact).
    norm = (rgb * (1.0 / (255.0 * STD)).astype(np.float32)
            + (-MEAN / STD).astype(np.float32)).transpose(2, 0, 1)[None]

    so = ort.SessionOptions()
    ref = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    tst = ort.InferenceSession(str(dest), so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input": norm.astype(np.float32)})
    tst_out = tst.run(None, {"input": img})
    for r, t, o in zip(ref_out, tst_out, ref.get_outputs()):
        d = float(np.abs(r - t).max())
        print(f"  {o.name}: max |diff| = {d:.2e}")
        assert d == 0.0, f"{o.name} diverged (expect bit-exact vs folded normalize)"
    print("OK: uint8 graph is BIT-EXACT vs the C++ folded preprocessing")


if __name__ == "__main__":
    main()
