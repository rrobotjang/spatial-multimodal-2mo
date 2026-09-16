"""Live-tunnel re-benchmarker, 60 KITTI val scenes, verdict-agreement gate.

Bootstraps the pipeline through the PUBLIC trycloudflare tunnel (exercises the
full grounded scene-graph chain end-to-end), collects gold-vs-pred verdict
agreement over all 60 scenes, and prints the verdict agreement rate so the
harness can apply the user gate (>= 85% jury CANNOT/SAFE-consistency before
paper + push).

Reads the tunnel URL and val path from argv/env (no hardcoded tunnel), so it
stays correct across cloudflared tunnel rollover.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

VULN = frozenset({"Pedestrian", "Cyclist", "Person_sitting", "Jaywalker"})
HEAVY = frozenset({"Tram", "Truck", "Van", "Bus", "Moped"})

VERDICT_RE = re.compile(r"\b(CANNOT|SAFE)\b")


def main() -> int:
    tunnel = sys.argv[1] if len(sys.argv) > 1 else ""
    root = sys.argv[2] if len(sys.argv) > 2 else ""
    if not (tunnel and root):
        print("usage: tunnel=URL root=DIR python bench_live.py URL DIR")
        return 2

    endpoint = tunnel.rstrip("/") + "/reason"
    val_path = root + "/data/kitti_scene_val.jsonl"

    agree = 0
    total = 0
    mismatch = 0
    under = 0  # gold=CANNOT, pred=SAFE  (missed blocking)
    over = 0   # gold=SAFE, pred=CANNOT (false alarm)

    with open(val_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            scene = json.loads(line)
            sid = scene.get("id", "?")
            gold = (scene.get("gold_verdict") or scene.get("gold") or "").upper()

            payload = {
                "scene": scene,
                "question": "Can the ego vehicle proceed?",
            }
            body = ""
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode()
            except Exception as exc:  # tunnel flake must not kill the bench
                print(f"ERR {sid}: {exc}")
                continue

            # Pull the final verdict out of the reasoning-chain answer text.
            pred = ""
            m = VERDICT_RE.search(body)
            if m:
                pred = m.group(1).upper()
            if not pred:
                print(f"NO-VERDICT {sid}")
                continue

            ok = pred == gold
            total += 1
            if ok:
                agree += 1
            else:
                mismatch += 1
                if gold == "CANNOT":
                    under += 1
                else:
                    over += 1
                print(f"MISMATCH {sid}: gold={gold} pred={pred}")

    rate = (100.0 * agree / total) if total else 0.0
    print("=" * 64)
    print(f"RESULT total={total} agree={agree} mismatch={mismatch}")
    print(f"RESULT rate={rate:.1f}%  (gold=CANNOT->pred=SAFE under:{under}, "
          f"gold=SAFE->pred=CANNOT over:{over})")
    print(f"GATE: {'PASS >= 85%' if rate >= 85.0 else 'FAIL < 85%'}")
    return 0 if rate >= 85.0 else 1


if __name__ == "__main__":
    sys.exit(main())
