#!/usr/bin/env python3
"""Phase E — smoke test for the running API (does NOT start the server).

    uvicorn api.main:app --host 0.0.0.0 --port 8000     # in another shell
    python scripts/10_smoke_api.py [--base-url http://localhost:8000]

Checks, in order: /health is ok; POST /analyze on a known TEST image returns
200 with the true genus printed next to the top-1 and an atlas_position that
lands near the specimen's own committed UMAP coordinate; the three error
paths (wrong content type -> 400, unreadable image -> 422, oversize -> 413);
GET /genera (with atlas medians), /atlas (all rows, aligned with
umap_coords.csv), /geo/{genus}, /images/{code} (a similar specimen from the
/analyze answer), /geo/unknown -> 404. Then /analyze latency, median of 5
calls on the same image. Any failed assertion exits non-zero.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
N_LATENCY_CALLS = 5
ATLAS_TOLERANCE = 1.0  # UMAP.transform is approximate; fitted coords span ~[-6, 12]

log = logging.getLogger("smoke")


def check(cond: bool, msg: str) -> None:
    if not cond:
        log.error("FAIL: %s", msg)
        sys.exit(1)
    log.info("ok   %s", msg)


def post_image(base: str, path: Path, content_type: str = "image/jpeg",
               data: bytes | None = None) -> requests.Response:
    payload = path.read_bytes() if data is None else data
    return requests.post(f"{base}/analyze", timeout=120,
                         files={"file": (path.name, payload, content_type)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--specimen", default=None, help="specimen_code of a test image")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    setup_logging(REPORTS / "10_smoke_api.log")

    dataset = pd.read_csv(DATA / "dataset.csv")
    test = dataset[dataset["split"] == "test"].sort_values("specimen_code")
    row = test[test["specimen_code"] == args.specimen].iloc[0] if args.specimen \
        else test.iloc[0]
    image = ROOT / row["image_path"]
    log.info("base url %s; test image %s (%s %s)", base, row["image_path"],
             row["genus"], row["species"])

    # /health
    r = requests.get(f"{base}/health", timeout=30)
    check(r.status_code == 200 and r.json()["status"] == "ok",
          f"GET /health -> {r.status_code} {r.json()}")
    health = r.json()
    check(health["model_loaded"] and health["n_embeddings"] == len(dataset),
          f"model loaded, n_embeddings {health['n_embeddings']} == dataset {len(dataset)}")

    # /analyze on a known test image
    t0 = time.perf_counter()
    r = post_image(base, image)
    dt = time.perf_counter() - t0
    check(r.status_code == 200, f"POST /analyze -> {r.status_code} in {dt * 1000:.0f} ms")
    body = r.json()
    top = body["predictions"][0]
    log.info("     true genus %-14s top-1 %-14s p=%.3f  %s",
             row["genus"], top["genus"], top["probability"],
             "MATCH" if top["genus"] == row["genus"] else "MISS")
    for p in body["predictions"]:
        log.info("       %-14s %-14s %.3f", p["genus"], p["subfamily"], p["probability"])
    check(len(body["predictions"]) == 3 and len(body["similar"]) == 5,
          "3 predictions and 5 similar specimens")
    codes = {s["specimen_code"] for s in body["similar"]}
    train_codes = set(dataset.loc[dataset["split"] == "train", "specimen_code"])
    check(codes <= train_codes and row["specimen_code"] not in codes,
          "similar specimens are TRAIN rows and exclude the query")
    for s in body["similar"]:
        log.info("       %s %-14s %-28s sim %.3f  %s / %s", s["specimen_code"], s["genus"],
                 s["species"] or "-", s["similarity"], s["photographer"], s["license"][:22])
    check(body["model_name"] and body["probe_version"],
          f"model {body['model_name']}, probe {body['probe_version']}")
    coords = pd.read_csv(DATA / "umap_coords.csv").set_index("specimen_code")
    own = coords.loc[row["specimen_code"]]
    pos = body["atlas_position"]
    check(pos is not None and all(k in pos for k in ("x", "y")), f"atlas_position {pos}")
    dist = ((pos["x"] - own["x"]) ** 2 + (pos["y"] - own["y"]) ** 2) ** 0.5
    check(dist < ATLAS_TOLERANCE,
          f"atlas_position ({pos['x']:.2f}, {pos['y']:.2f}) is {dist:.2f} from the fitted "
          f"({own['x']:.2f}, {own['y']:.2f})")

    # error paths
    r = post_image(base, image, content_type="text/plain")
    check(r.status_code == 400 and "message" in r.json(), f"text/plain -> 400 {r.json()}")
    r = post_image(base, image, data=b"definitely not a jpeg")
    check(r.status_code == 422 and "message" in r.json(), f"garbage -> 422 {r.json()}")
    r = post_image(base, image, data=b"\xff" * (11 << 20))
    check(r.status_code == 413 and "message" in r.json(), f"11 MB -> 413 {r.json()}")

    # /genera
    r = requests.get(f"{base}/genera", timeout=30)
    check(r.status_code == 200 and r.json()["n"] == dataset["genus"].nunique(),
          f"GET /genera -> {r.status_code}, {r.json()['n']} genera")
    g0 = r.json()["genera"][0]
    log.info("     e.g. %s", g0)
    check(all("atlas_median" in g and {"x", "y"} <= set(g["atlas_median"])
              for g in r.json()["genera"]), "every genus has an atlas_median")

    # /atlas
    t0 = time.perf_counter()
    r = requests.get(f"{base}/atlas", timeout=60)
    dt = time.perf_counter() - t0
    check(r.status_code == 200, f"GET /atlas -> {r.status_code}, {len(r.content) / 1024:.0f} KB "
                                f"in {dt * 1000:.0f} ms")
    atlas = r.json()
    check(atlas["n"] == len(coords) == len(atlas["points"]),
          f"atlas has {atlas['n']} points == umap_coords.csv rows")
    pts = pd.DataFrame(atlas["points"]).set_index("specimen_code")
    check(list(pts.index) == list(coords.index)
          and (pts[["x", "y"]] - coords[["x", "y"]]).abs().max().max() < 1e-3,
          "atlas rows and coordinates match umap_coords.csv")
    check(set(pts.columns) >= {"x", "y", "genus", "subfamily", "species", "image_available"}
          and pts["image_available"].all(), "atlas fields present, every image available")
    log.info("     umap params %s; e.g. %s", atlas["umap"], atlas["points"][0])
    t0 = time.perf_counter()
    r = requests.get(f"{base}/atlas", timeout=60)
    log.info("     second (cached) call %.0f ms", (time.perf_counter() - t0) * 1000)

    # /geo
    r = requests.get(f"{base}/geo/{top['genus']}", timeout=30)
    check(r.status_code == 200, f"GET /geo/{top['genus']} -> {r.status_code}")
    geo = r.json()
    log.info("     %d specimens, provinces %s, elev %s, years %s-%s, %d points%s",
             geo["n_specimens"], geo["provinces"], geo["elevation"], geo["year_min"],
             geo["year_max"], len(geo["points"]), " (capped)" if geo["points_capped"] else "")
    r = requests.get(f"{base}/geo/Notarealgenus", timeout=30)
    check(r.status_code == 404 and "message" in r.json(), "GET /geo/Notarealgenus -> 404")

    # /images
    code = body["similar"][0]["specimen_code"]
    r = requests.get(body["similar"][0]["image_url"], timeout=30)
    check(r.status_code == 200 and r.headers["content-type"] == "image/jpeg",
          f"GET {body['similar'][0]['image_url']} -> {r.status_code}, {len(r.content)} bytes")
    r = requests.get(f"{base}/images/../config.yaml", timeout=30)
    check(r.status_code == 404, "GET /images/../config.yaml -> 404")

    # latency
    times = []
    for _ in range(N_LATENCY_CALLS):
        t0 = time.perf_counter()
        r = post_image(base, image)
        times.append((time.perf_counter() - t0) * 1000)
        check(r.status_code == 200, f"latency call -> {r.status_code} in {times[-1]:.0f} ms")
    log.info("/analyze latency: median %.0f ms, min %.0f, max %.0f (%d calls, "
             "embed only %.0f ms)", statistics.median(times), min(times), max(times),
             len(times), r.json()["embed_ms"])
    log.info("smoke test passed")


if __name__ == "__main__":
    main()
