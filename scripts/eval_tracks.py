#!/usr/bin/env python3
"""Objective A/B comparison of tracker configs from locked-target JSONL.

No ground-truth labels needed. Run the SAME clip through each tracker config with
--autolock and --out, then compare the resulting logs:

    VID=path/to/clip.mp4
    ./build/object_tracking $VID configs/rfdetr_medium_sahi.yaml         --autolock --out out/bytetrack.jsonl
    ./build/object_tracking $VID configs/rfdetr_medium_sahi_ocsort.yaml  --autolock --out out/ocsort.jsonl
    ./build/object_tracking $VID configs/rfdetr_medium_sahi_botsort.yaml --autolock --out out/botsort.jsonl
    python3 scripts/eval_tracks.py out/bytetrack.jsonl out/ocsort.jsonl out/botsort.jsonl

For a fair comparison use the same clip, the same --start, and --autolock so each
run locks the same central target. Real-time frame-dropping means a slower tracker
processes fewer frames, so all rate metrics are normalized per-frame and `coverage`
(fraction of source frames actually processed) is itself a cost signal.

Metrics and which of the four axes each one speaks to:

  id_switch_rate   (Hold lock on crossings/pans) raw MOT track_id changes while the
                   lock_id stays fixed, per 1000 logged frames. The cleanest tracker-
                   quality signal: each switch = the MOT handed the lock a new id for
                   the same target, which the lock then had to paper over. LOWER better.
  coasting_per_min (Hold lock)        locked -> coasting transitions (brief track loss). LOWER better.
  lost_per_min     (Recover)          entries into LOST (target dropped). LOWER better.
  lock_continuity  (Recover/Hold)     fraction of logged frames actually Locked.        HIGHER better.
  jitter_px        (Smoothness)       RMS of locked-box center acceleration, px/frame^2,
                   over consecutive frames. LOWER = smoother box. (Same clip across
                   trackers, so the delta is the tracker/Kalman, not the real motion.)
  coverage         (Cost)             logged frames / source-frame span processed.       HIGHER = faster.
"""
import argparse
import json
import math
import os
import sys

# (key, label, better_direction)  better: -1 lower is better, +1 higher is better
METRICS = [
    ("id_switch_rate",   "id switches /1k fr",   -1),
    ("coasting_per_min", "coasting /min",        -1),
    ("lost_per_min",     "lost /min",            -1),
    ("lock_continuity",  "lock continuity",      +1),
    ("jitter_px",        "jitter px/fr^2",       -1),
    ("coverage",         "frame coverage",       +1),
]

LOCKED_STATES = {"locked", "coasting"}


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recs.sort(key=lambda r: r.get("frame", 0))
    return recs


def analyze(recs):
    if not recs:
        return None
    frames = [r.get("frame", 0) for r in recs]
    span = (max(frames) - min(frames) + 1) if len(frames) > 1 else 1
    n = len(recs)

    # Source frame rate: timestamp_s = frame/fps, so fps = 1 / median(dt/dframe).
    fps = 30.0
    deltas = []
    for a, b in zip(recs, recs[1:]):
        df = b.get("frame", 0) - a.get("frame", 0)
        dt = b.get("t", 0.0) - a.get("t", 0.0)
        if df > 0 and dt > 0:
            deltas.append(dt / df)
    if deltas:
        deltas.sort()
        med = deltas[len(deltas) // 2]
        if med > 0:
            fps = 1.0 / med
    duration_min = (span / fps) / 60.0 if fps > 0 else 0.0

    locked = sum(1 for r in recs if r.get("state") == "locked")
    id_switches = 0
    coasting_episodes = 0
    lost_episodes = 0
    accel_sq = []
    prev = None
    prev_vel = None  # (vx, vy, frame) of last step, for acceleration
    for r in recs:
        st = r.get("state")
        if prev is not None:
            pst = prev.get("state")
            # raw MOT id flip while the same lock is held = association switch
            if (r.get("lock_id", -1) >= 0 and r.get("lock_id") == prev.get("lock_id")
                    and r.get("track_id", -1) >= 0 and prev.get("track_id", -1) >= 0
                    and r.get("track_id") != prev.get("track_id")):
                id_switches += 1
            if pst != "coasting" and st == "coasting":
                coasting_episodes += 1
            if pst != "lost" and st == "lost":
                lost_episodes += 1
        # jitter: acceleration of center over consecutive close frames while held
        if st in LOCKED_STATES and prev is not None and prev.get("state") in LOCKED_STATES:
            df = r.get("frame", 0) - prev.get("frame", 0)
            cpx = r.get("center_px", [0, 0])
            ppx = prev.get("center_px", [0, 0])
            if 0 < df <= 3:
                vx = (cpx[0] - ppx[0]) / df
                vy = (cpx[1] - ppx[1]) / df
                if prev_vel is not None:
                    pvx, pvy, pf = prev_vel
                    adf = r.get("frame", 0) - pf
                    if 0 < adf <= 3:
                        ax = (vx - pvx) / adf
                        ay = (vy - pvy) / adf
                        accel_sq.append(ax * ax + ay * ay)
                prev_vel = (vx, vy, r.get("frame", 0))
            else:
                prev_vel = None
        elif st not in LOCKED_STATES:
            prev_vel = None
        prev = r

    held = sum(1 for r in recs if r.get("state") in LOCKED_STATES)
    jitter = math.sqrt(sum(accel_sq) / len(accel_sq)) if accel_sq else 0.0
    return {
        "n": n,
        "fps": fps,
        "duration_min": duration_min,
        "id_switch_rate": 1000.0 * id_switches / held if held else 0.0,
        "id_switches": id_switches,
        "coasting_per_min": coasting_episodes / duration_min if duration_min else 0.0,
        "lost_per_min": lost_episodes / duration_min if duration_min else 0.0,
        "lock_continuity": locked / n if n else 0.0,
        "jitter_px": jitter,
        "coverage": n / span if span else 0.0,
    }


def fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="locked-target JSONL files (one per tracker config)")
    args = ap.parse_args()

    cols = []
    for path in args.logs:
        recs = load(path)
        a = analyze(recs)
        name = os.path.splitext(os.path.basename(path))[0]
        if a is None:
            print(f"warning: {path} is empty or unreadable — skipping", file=sys.stderr)
            continue
        cols.append((name, a))

    if not cols:
        print("no usable logs", file=sys.stderr)
        return 1

    label_w = max(len("metric"), max(len(l) for _, l, _ in METRICS)) + 1
    col_w = max(12, max(len(name) for name, _ in cols) + 1)

    header = "metric".ljust(label_w) + "".join(name.rjust(col_w) for name, _ in cols)
    print(header)
    print("-" * len(header))
    # context rows
    for key, label in [("n", "frames logged"), ("duration_min", "duration (min)")]:
        row = label.ljust(label_w) + "".join(fmt(a[key]).rjust(col_w) for _, a in cols)
        print(row)
    print("-" * len(header))

    for key, label, better in METRICS:
        vals = [a[key] for _, a in cols]
        best = (min if better < 0 else max)(vals)
        cells = ""
        for v in vals:
            s = fmt(v)
            if len(cols) > 1 and abs(v - best) < 1e-9:
                s = s + "*"          # mark the best column
            cells += s.rjust(col_w)
        arrow = "↓" if better < 0 else "↑"
        print(f"{label} {arrow}".ljust(label_w) + cells)

    print("-" * len(header))
    print("* = best in row.  ↓ lower is better, ↑ higher is better.")
    print("id switches = raw MOT track_id flips under a held lock_id (the core tracker-quality signal).")
    print("coverage<1 or low fps in one column = that tracker is dropping frames (too slow for real time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
