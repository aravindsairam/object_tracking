#!/usr/bin/env python3
"""Backend-parity comparison for parity_dump video-mode JSONL files.

Compares a test dump (e.g. TRT fp16) against a reference dump (e.g. ORT CUDA EP)
frame by frame with box-aware metrics — the D-FINE lesson: score-only parity is
blind to DETR box drift, so we match boxes per class by IoU and report drift,
misses and hallucinations, plus near-threshold flips (a det that merely crossed
the conf cut between backends is bookkept separately from a true disagreement).

  python3 tools/parity_compare.py --ref out/parity/cuda_ep_0001.jsonl \
      --test out/parity/trt_fp16_0001.jsonl [--conf 0.3] [--iou 0.5] \
      [--video clip.MP4 --outdir out/parity/vis_0001 --worst 8]
"""
import argparse
import json
import os


def load(path):
    frames = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            frames[r["frame"]] = r
    return frames


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(ref, test, iou_thr):
    """Greedy same-class IoU matching, highest IoU first. Returns
    (pairs [(ri, ti, iou)], ref_unmatched idxs, test_unmatched idxs)."""
    cands = []
    for ri, r in enumerate(ref):
        for ti, t in enumerate(test):
            if r["c"] != t["c"]:
                continue
            v = iou(r["b"], t["b"])
            if v >= iou_thr:
                cands.append((v, ri, ti))
    cands.sort(reverse=True)
    used_r, used_t, pairs = set(), set(), []
    for v, ri, ti in cands:
        if ri in used_r or ti in used_t:
            continue
        used_r.add(ri); used_t.add(ti)
        pairs.append((ri, ti, v))
    return (pairs,
            [i for i in range(len(ref)) if i not in used_r],
            [i for i in range(len(test)) if i not in used_t])


def pct(x, n):
    return f"{100.0 * x / n:.2f}%" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--conf", type=float, default=0.3, help="production threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="match IoU gate")
    ap.add_argument("--video", help="source clip, for rendering disagreement frames")
    ap.add_argument("--outdir", help="where to write annotated PNGs")
    ap.add_argument("--worst", type=int, default=8, help="render N worst + 2 control frames")
    ap.add_argument("--json-out", help="write summary JSON here")
    a = ap.parse_args()

    ref_frames, test_frames = load(a.ref), load(a.test)
    common = sorted(set(ref_frames) & set(test_frames))
    if not common:
        raise SystemExit("no common frames between dumps")

    tot = dict(ref=0, test=0, matched=0, ref_only=0, test_only=0,
               ref_flip=0, test_flip=0)
    ious, sdeltas, cdrifts = [], [], []
    per_frame = []  # (disagreement score, frame, detail)

    for fr in common:
        R, T = ref_frames[fr], test_frames[fr]
        # production view: dets above conf; sub-threshold pool for flip analysis
        rp = [d for d in R["dets"] if d["s"] >= a.conf]
        tp = [d for d in T["dets"] if d["s"] >= a.conf]
        rs = [d for d in R["dets"] if d["s"] < a.conf]
        ts = [d for d in T["dets"] if d["s"] < a.conf]

        pairs, r_un, t_un = greedy_match(rp, tp, a.iou)
        tot["ref"] += len(rp); tot["test"] += len(tp); tot["matched"] += len(pairs)

        for ri, ti, v in pairs:
            ious.append(v)
            sdeltas.append(abs(rp[ri]["s"] - tp[ti]["s"]))
            rb, tb = rp[ri]["b"], tp[ti]["b"]
            cdrifts.append(((rb[0] + rb[2] / 2 - tb[0] - tb[2] / 2) ** 2 +
                            (rb[1] + rb[3] / 2 - tb[1] - tb[3] / 2) ** 2) ** 0.5)

        # unmatched ref det: did test see it below conf? -> threshold flip
        r_true, t_true = [], []
        for i in r_un:
            fp, _, _ = greedy_match([rp[i]], ts, a.iou)
            if fp: tot["ref_flip"] += 1
            else:  tot["ref_only"] += 1; r_true.append(rp[i])
        for i in t_un:
            fp, _, _ = greedy_match([tp[i]], rs, a.iou)
            if fp: tot["test_flip"] += 1
            else:  tot["test_only"] += 1; t_true.append(tp[i])

        f_iou = [v for _, _, v in pairs]
        dis = len(r_true) + len(t_true) + sum(1.0 - v for v in f_iou)
        per_frame.append((dis, fr, dict(pairs=pairs, rp=rp, tp=tp,
                                        r_true=r_true, t_true=t_true)))

    n = len(common)
    ious.sort()
    def q(lst, p):
        return lst[int(p * (len(lst) - 1))] if lst else float("nan")

    print(f"parity: test={os.path.basename(a.test)} vs ref={os.path.basename(a.ref)}")
    print(f"  frames={n}  conf={a.conf}  match_iou>={a.iou}")
    print(f"  dets/frame: ref {tot['ref']/n:.1f}  test {tot['test']/n:.1f}")
    print(f"  matched:        {tot['matched']:5d}  ({pct(tot['matched'], tot['ref'])} of ref)")
    print(f"  MISSED by test: {tot['ref_only']:5d}  ({pct(tot['ref_only'], tot['ref'])})   "
          f"[+{tot['ref_flip']} threshold flips]")
    print(f"  EXTRA in test:  {tot['test_only']:5d}  ({pct(tot['test_only'], tot['test'])})   "
          f"[+{tot['test_flip']} threshold flips]")
    if ious:
        print(f"  matched IoU: mean {sum(ious)/len(ious):.4f}  median {q(ious,0.5):.4f}  "
              f"p5 {q(ious,0.05):.4f}  min {ious[0]:.4f}")
        print(f"  score |delta|: mean {sum(sdeltas)/len(sdeltas):.4f}  max {max(sdeltas):.4f}")
        print(f"  center drift px: mean {sum(cdrifts)/len(cdrifts):.2f}  max {max(cdrifts):.2f}")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(dict(frames=n, **tot,
                           iou_mean=sum(ious)/len(ious) if ious else None,
                           iou_p5=q(ious, 0.05), iou_min=ious[0] if ious else None,
                           score_mae=sum(sdeltas)/len(sdeltas) if sdeltas else None,
                           drift_mean_px=sum(cdrifts)/len(cdrifts) if cdrifts else None),
                      f, indent=2)

    # ---- visual check: render the worst-disagreement frames + 2 controls ----
    if a.video and a.outdir:
        import cv2
        os.makedirs(a.outdir, exist_ok=True)
        per_frame.sort(reverse=True, key=lambda x: x[0])
        want = {fr: (dis, det) for dis, fr, det in per_frame[:a.worst]}
        for dis, fr, det in sorted(per_frame, key=lambda x: x[0])[:2]:  # controls
            want.setdefault(fr, (dis, det))

        cap = cv2.VideoCapture(a.video)
        wh = ref_frames[common[0]]["h"]
        idx, got = -1, 0
        while got < len(want):
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if idx not in want:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if frame.shape[0] > wh:  # replicate the tool's integer-math resize
                w = frame.shape[1] * wh // frame.shape[0]
                frame = cv2.resize(frame, (w, wh), interpolation=cv2.INTER_AREA)
            dis, d = want[idx]
            for r in d["rp"]:   # reference: green
                x, y, w, h = map(int, r["b"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            for t in d["tp"]:   # test: red, thin — coincident boxes ring the green
                x, y, w, h = map(int, t["b"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            for r in d["r_true"]:  # true misses: filled marker + score
                x, y, w, h = map(int, r["b"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"MISS {r['s']:.2f}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            for t in d["t_true"]:  # hallucinations
                x, y, w, h = map(int, t["b"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(frame, f"EXTRA {t['s']:.2f}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            cv2.putText(frame, f"frame {idx}  disagreement {dis:.2f}  "
                        f"green=ref red=test yellow=miss magenta=extra",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(a.outdir, f"frame_{idx:06d}.png"), frame)
            got += 1
        cap.release()
        print(f"  rendered {got} frames -> {a.outdir}")


if __name__ == "__main__":
    main()
