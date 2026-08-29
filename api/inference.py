"""Pure inference helpers: no FastAPI imports, no global state.

embed_image()    PIL image -> L2-normalised BioCLIP 2 vector (768,)
predict_genus()  vector -> top-k linear-probe (genus, subfamily, probability)
find_similar()   vector -> top-k cosine neighbours among the train embeddings
atlas_position() vector -> (x, y) on the fitted UMAP, or None if transform fails

The embedding convention is the one of scripts/06_embed.py (same preprocess,
same normalisation) so query vectors are comparable with data/embeddings.npy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("api")


@dataclass(frozen=True)
class Neighbour:
    row: int          # row of the train index / embedding matrix
    similarity: float


def embed_image(model, preprocess, image: Image.Image) -> np.ndarray:
    """Return the L2-normalised float32 embedding of one RGB PIL image."""
    with torch.no_grad():
        feats = model.encode_image(preprocess(image).unsqueeze(0))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].to(torch.float32).cpu().numpy()


def predict_genus(clf, embedding: np.ndarray, subfamily_of: Mapping[str, str],
                  k: int = 3) -> list[tuple[str, str, float]]:
    """Top-k (genus, subfamily, probability) from the logistic-regression probe."""
    proba = clf.predict_proba(embedding[None, :])[0]
    order = np.argsort(-proba, kind="stable")[:k]
    return [(str(clf.classes_[i]), subfamily_of[str(clf.classes_[i])], float(proba[i]))
            for i in order]


def find_similar(embedding: np.ndarray, train_embeddings: np.ndarray,
                 k: int = 5) -> list[Neighbour]:
    """Top-k cosine neighbours (rows of train_embeddings, descending)."""
    sims = train_embeddings @ embedding
    order = np.argsort(-sims, kind="stable")[:k]
    return [Neighbour(int(i), float(sims[i])) for i in order]


def atlas_position(reducer, embedding: np.ndarray) -> tuple[float, float] | None:
    """Project one embedding onto the fitted UMAP (data/umap_model.pkl).
    UMAP.transform is approximate (a training row lands near, not on, its
    fitted coordinate) and can fail on degenerate input; failures give None
    and the caller logs them."""
    try:
        xy = reducer.transform(embedding[None, :].astype(np.float32))[0]
    except Exception as exc:  # noqa: BLE001 (numba/umap raise assorted types)
        log.warning("umap transform failed: %s", exc)
        return None
    if not np.all(np.isfinite(xy)):
        log.warning("umap transform returned non-finite %s", xy)
        return None
    return float(xy[0]), float(xy[1])
