"""Pydantic response models for the mg-ants API (one per route)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    message: str


class GenusPrediction(BaseModel):
    genus: str
    subfamily: str
    probability: float = Field(ge=0.0, le=1.0)


class SimilarSpecimen(BaseModel):
    specimen_code: str
    genus: str
    subfamily: str
    species: str | None = None
    similarity: float = Field(description="cosine similarity in BioCLIP 2 space")
    image_url: str = Field(description="this API's /images/{specimen_code} route")
    antweb_url: str
    photographer: str | None = None
    license: str | None = None
    locality: str | None = None


class AtlasPosition(BaseModel):
    x: float
    y: float


class AnalyzeResponse(BaseModel):
    predictions: list[GenusPrediction] = Field(description="linear-probe top-3")
    similar: list[SimilarSpecimen] = Field(description="5 nearest train specimens")
    atlas_position: AtlasPosition | None = Field(
        description="query placed on the UMAP atlas (umap_model.transform); null if it failed")
    model_name: str
    probe_version: str = Field(description="mtime of data/probe.pkl, ISO-8601")
    embed_ms: float
    filename: str | None = None


class GenusInfo(BaseModel):
    name: str
    subfamily: str
    n_train: int
    n_test: int
    f1_probe: float
    atlas_median: AtlasPosition = Field(description="median UMAP position of the genus")


class GeneraResponse(BaseModel):
    genera: list[GenusInfo]
    n: int


class ElevationStats(BaseModel):
    min: float | None = None
    median: float | None = None
    max: float | None = None
    n: int


class GeoResponse(BaseModel):
    genus: str
    subfamily: str
    n_specimens: int
    n_species: int = Field(description="named species (geo_summary.csv)")
    n_unidentified: int = Field(description="specimens identified to genus only")
    provinces: dict[str, int] = Field(description="normalised province -> count")
    elevation: ElevationStats
    year_min: int | None = None
    year_max: int | None = None
    n_with_coords: int
    points: list[tuple[float, float]] = Field(description="[lat, lon]; capped")
    points_capped: bool


class AtlasPoint(BaseModel):
    specimen_code: str
    x: float
    y: float
    genus: str
    subfamily: str
    species: str | None = None
    image_available: bool


class AtlasResponse(BaseModel):
    points: list[AtlasPoint]
    n: int
    umap: dict[str, float | int | str] = Field(description="fit parameters, for the axes caption")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_embeddings: int
    n_train: int
    n_genera: int
    versions: dict[str, str]
