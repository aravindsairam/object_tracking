#!/usr/bin/env python3
"""COCO mAP for parity_dump coco-mode JSONL dumps (Tier 2 of the accuracy audit).

The dump already contains RF-DETR's official postprocess (top-K over Q*C sigmoid
scores, class = 91-slot COCO category id), so the number here is directly
comparable to the paper's RF-DETR-Medium reference (54.7 AP50:95, TRT fp16 b1).

  python3 tools/coco_eval_dump.py --ann .../instances_val2017.json \
      --dump out/parity/coco_trt_fp16.jsonl [--dump more.jsonl ...]
"""
import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(coco, dump_path):
    results = []
    n_img = 0
    with open(dump_path) as f:
        for line in f:
            r = json.loads(line)
            n_img += 1
            for c, s, x, y, w, h in r["dets"]:
                results.append(dict(image_id=r["image_id"], category_id=int(c),
                                    bbox=[x, y, w, h], score=s))
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.imgIds = sorted({d["image_id"] for d in results})
    ev.evaluate()
    ev.accumulate()
    print(f"\n===== {dump_path}  ({n_img} images) =====")
    ev.summarize()
    return ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--dump", action="append", required=True)
    a = ap.parse_args()

    coco = COCO(a.ann)
    rows = [(d, evaluate(coco, d)) for d in a.dump]

    print(f"\n{'dump':40s} {'AP':>6s} {'AP50':>6s} {'AP75':>6s} {'APs':>6s} {'APm':>6s} {'APl':>6s}")
    for d, s in rows:
        name = d.split("/")[-1]
        print(f"{name:40s} {s[0]*100:6.2f} {s[1]*100:6.2f} {s[2]*100:6.2f} "
              f"{s[3]*100:6.2f} {s[4]*100:6.2f} {s[5]*100:6.2f}")


if __name__ == "__main__":
    main()
