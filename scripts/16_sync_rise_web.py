#!/usr/bin/env python3
"""Publish the RISE outputs to the web app (replaces the manual copy).

1. Copies every PNG in reports/rise/web/ (overlays, shared-scale overlays,
   letterboxed photos written by 12_rise.py) to web/public/rise/.
2. Updates the numbers in web/src/data/rise.json from reports/rise/runs.json:
   per run pFull and baseline, and the run settings n_masks, grid, p_keep,
   probs, shared_w. Captions, species, creator and row order in rise.json are
   hand-written and kept as they are. A run listed in rise.json but missing
   from runs.json is an error.

--check writes nothing and exits 1 if anything would change, so it doubles
as the test that the published files match the RISE outputs.
Offline; run after scripts/12_rise.py, then rebuild the web app.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "rise" / "web"
RUNS = ROOT / "reports" / "rise" / "runs.json"
DEST = ROOT / "web" / "public" / "rise"
WEB_JSON = ROOT / "web" / "src" / "data" / "rise.json"
SETTINGS = ("n_masks", "grid", "p_keep", "probs", "shared_w")


def synced_json(current: dict, runs: dict) -> dict:
    out = json.loads(json.dumps(current))
    by_key = {(r["specimen_code"], r["target"]): r for r in runs["runs"]}
    for r in out["runs"]:
        key = (r["code"], r["target"])
        if key not in by_key:
            raise KeyError(f"rise.json lists {key}, which is not in {RUNS}")
        src = by_key[key]
        r["pFull"] = src["p_full"]
        r["baseline"] = src["baseline"]
    for k in SETTINGS:
        out[k] = runs[k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report differences, write nothing")
    parser.add_argument("--dest", type=Path, default=DEST, help=f"image folder (default {DEST})")
    args = parser.parse_args()

    pngs = sorted(SRC.glob("*.png"))
    if not pngs:
        sys.exit(f"no PNGs in {SRC}; run scripts/12_rise.py first")
    changed = [p for p in pngs if not (args.dest / p.name).exists()
               or not filecmp.cmp(p, args.dest / p.name, shallow=False)]
    extra = sorted({p.name for p in args.dest.glob("*.png")} - {p.name for p in pngs}) \
        if args.dest.exists() else []

    runs = json.loads(RUNS.read_text())
    current_text = WEB_JSON.read_text()
    new_text = json.dumps(synced_json(json.loads(current_text), runs), indent=2, ensure_ascii=False) + "\n"
    json_changes = new_text != current_text

    print(f"{len(pngs)} PNGs in {SRC.relative_to(ROOT)}; {len(changed)} new or different in {args.dest}")
    for p in changed:
        print("  ", p.name)
    if extra:
        print(f"{len(extra)} PNGs in {args.dest} not produced by 12_rise.py (left in place): {extra}")
    print(f"{WEB_JSON.relative_to(ROOT)}: {'would change' if json_changes else 'up to date'}")

    if args.check:
        sys.exit(1 if changed or json_changes else 0)
    args.dest.mkdir(parents=True, exist_ok=True)
    for p in changed:
        shutil.copy2(p, args.dest / p.name)
    if json_changes:
        WEB_JSON.write_text(new_text)
    print("synced" if changed or json_changes else "nothing to do")


if __name__ == "__main__":
    main()
