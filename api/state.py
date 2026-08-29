"""Everything the API needs in memory, loaded once at startup.

AppState.load() reads config.yaml and:
- the BioCLIP 2 image tower (open_clip) and its preprocess
- data/probe.pkl (LogisticRegression from 07_eval.py)
- data/embeddings.npy + embeddings_index.csv (rows asserted to match)
- data/dataset.csv (specimen metadata for the similar-specimen cards)
- data/dataset_full.csv (coordinates / provinces for /geo, provinces
  normalised with the same aliases as 09_geo.py)
- data/geo_summary.csv, data/geo_by_place.csv, reports/per_genus.csv
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from gbif_client import load_config, quiet_logging, setup_logging  # noqa: E402

PROBE_PATH = DATA / "probe.pkl"
EMB_PATH = DATA / "embeddings.npy"
INDEX_PATH = DATA / "embeddings_index.csv"
DATASET_META_COLS = ["specimen_code", "genus", "subfamily", "species", "locality",
                     "creator", "license", "image_path", "split"]

log = logging.getLogger("api")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) \
        .replace(microsecond=0).isoformat()


@dataclass
class AppState:
    config: dict[str, Any]
    model: Any
    preprocess: Any
    probe: Any
    probe_version: str
    embeddings: np.ndarray            # all rows, aligned with `index`
    index: pd.DataFrame               # embeddings_index.csv
    index_sha256: str
    train_embeddings: np.ndarray      # rows of `embeddings` where split == train
    train_index: pd.DataFrame         # matching rows, reset_index
    specimens: pd.DataFrame           # dataset.csv metadata, indexed by specimen_code
    full: pd.DataFrame                # dataset_full.csv with `province`
    geo_summary: pd.DataFrame         # indexed by genus
    geo_by_place: pd.DataFrame
    per_genus: pd.DataFrame           # indexed by genus
    subfamily_of: dict[str, str]
    load_seconds: float
    infer_lock: Lock = field(default_factory=Lock)

    @property
    def model_name(self) -> str:
        return str(self.config["embed_model"])

    @classmethod
    def load(cls) -> "AppState":
        setup_logging(REPORTS / "api.log")
        t0 = time.perf_counter()
        cfg = load_config(ROOT / "config.yaml")

        import open_clip  # slow import
        import joblib

        torch.set_num_threads(os.cpu_count() or 1)
        # huggingface_hub nags about a missing HF_TOKEN on every cached load
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        t = time.perf_counter()
        with quiet_logging():
            model, _, preprocess = open_clip.create_model_and_transforms(cfg["embed_model"])
        model.eval()
        log.info("model %s loaded in %.1fs (%d torch threads)",
                 cfg["embed_model"], time.perf_counter() - t, torch.get_num_threads())

        probe = joblib.load(PROBE_PATH)
        probe_version = _mtime_iso(PROBE_PATH)
        log.info("probe %s (%s, %d classes, version %s)",
                 PROBE_PATH.name, type(probe).__name__, len(probe.classes_), probe_version)

        embeddings = np.load(EMB_PATH)
        index = pd.read_csv(INDEX_PATH)
        assert embeddings.shape[0] == len(index), \
            f"embeddings rows {embeddings.shape[0]} != index rows {len(index)}"
        assert embeddings.dtype == np.float32
        assert index["specimen_code"].is_unique
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "embeddings are not L2-normalised"
        index_sha = _sha256(INDEX_PATH)
        train_mask = (index["split"] == "train").to_numpy()
        train_index = index[train_mask].reset_index(drop=True)
        train_embeddings = np.ascontiguousarray(embeddings[train_mask])
        log.info("embeddings %s (%d train, %d test), index sha256 %s",
                 embeddings.shape, train_mask.sum(), (~train_mask).sum(), index_sha[:16])

        specimens = pd.read_csv(DATA / "dataset.csv", usecols=DATASET_META_COLS)
        assert specimens["specimen_code"].is_unique
        specimens = specimens.set_index("specimen_code")
        missing = set(index["specimen_code"]) - set(specimens.index)
        assert not missing, f"{len(missing)} embedded specimens absent from dataset.csv"

        full = pd.read_csv(DATA / "dataset_full.csv",
                           usecols=["specimen_code", "genus", "subfamily", "stateProvince",
                                    "decimalLatitude", "decimalLongitude", "elevation",
                                    "eventDate"])
        full["province"] = full["stateProvince"].str.strip().replace(cfg["province_aliases"])
        full["year"] = pd.to_numeric(full["eventDate"].str[:4], errors="coerce")

        geo_summary = pd.read_csv(DATA / "geo_summary.csv").set_index("genus")
        geo_by_place = pd.read_csv(DATA / "geo_by_place.csv")
        per_genus = pd.read_csv(REPORTS / "per_genus.csv").set_index("genus")
        missing = set(probe.classes_) - set(per_genus.index)
        assert not missing, f"probe classes absent from per_genus.csv: {missing}"

        subfamily_of = (index.drop_duplicates("genus").set_index("genus")["subfamily"]
                        .to_dict())
        subfamily_of.update(full.drop_duplicates("genus").set_index("genus")["subfamily"]
                            .to_dict())
        log.info("tables: dataset %d, dataset_full %d (%d genera), geo_summary %d, "
                 "geo_by_place %d, per_genus %d",
                 len(specimens), len(full), full["genus"].nunique(), len(geo_summary),
                 len(geo_by_place), len(per_genus))

        load_seconds = time.perf_counter() - t0
        log.info("state loaded in %.1fs", load_seconds)
        return cls(config=cfg, model=model, preprocess=preprocess, probe=probe,
                   probe_version=probe_version, embeddings=embeddings, index=index,
                   index_sha256=index_sha, train_embeddings=train_embeddings,
                   train_index=train_index, specimens=specimens, full=full,
                   geo_summary=geo_summary, geo_by_place=geo_by_place,
                   per_genus=per_genus, subfamily_of=subfamily_of,
                   load_seconds=load_seconds)

    def versions(self) -> dict[str, str]:
        import fastapi
        import open_clip
        import sklearn
        return {
            "model_name": self.model_name,
            "probe_version": self.probe_version,
            "embeddings_index_sha256": self.index_sha256,
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "scikit_learn": sklearn.__version__,
            "fastapi": fastapi.__version__,
        }
