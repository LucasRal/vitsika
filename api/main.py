"""Phase E — FastAPI serving layer for the mg-ants genus classifier.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Routes
  POST /analyze                multipart image -> probe top-3 + 5 similar train specimens
                               + the query's position on the UMAP atlas
  GET  /genera                 the 27 POC genera with split sizes, probe F1, atlas median
  GET  /atlas                  every specimen's UMAP position (cached JSON)
  GET  /geo/{genus}            province counts, elevation, years, [lat, lon] points
  GET  /images/{specimen_code} the local profile-view jpg (thumbnails for the UI)
  GET  /health                 status, model_loaded, n_embeddings, versions

All errors are JSON bodies with a `message` field: 400 wrong content type,
413 upload too large, 422 unreadable image, 404 unknown genus / specimen.
Model, probe, embeddings and tables are loaded once in the lifespan
(api/state.py); inference itself lives in api/inference.py.
"""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from api.inference import atlas_position, embed_image, find_similar, predict_genus
from api.schemas import (AnalyzeResponse, AtlasPosition, AtlasResponse, ElevationStats,
                         ErrorResponse, GeneraResponse, GenusInfo, GenusPrediction,
                         GeoResponse, HealthResponse, SimilarSpecimen)
from api.state import ROOT, AppState

ANTWEB_SPECIMEN_URL = "https://www.antweb.org/specimen/{code}"
READ_CHUNK = 1 << 20
_ERROR_RESPONSES = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                    422: {"model": ErrorResponse}}

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ctx = await run_in_threadpool(AppState.load)
    yield
    log.info("shutdown")


app = FastAPI(title="mg-ants", version="0.1",
              description="Malagasy ant genus classifier (BioCLIP 2 + linear probe)",
              lifespan=lifespan)


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"message": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()))
    return JSONResponse({"message": f"invalid request: {where}: {first.get('msg', exc)}"},
                        status_code=422)


def _clean(value) -> str | None:
    """pandas NaN -> None for optional string fields."""
    return None if value is None or value != value else str(value)


# ------------------------------------------------------------------- /analyze
async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    if not (file.content_type or "").lower().startswith("image/"):
        raise HTTPException(400, f"expected an image/* upload, got "
                                 f"{file.content_type or 'no content type'}")
    buf = bytearray()
    while chunk := await file.read(READ_CHUNK):
        buf += chunk
        if len(buf) > max_bytes:
            raise HTTPException(413, f"image larger than {max_bytes // (1 << 20)} MB")
    if not buf:
        raise HTTPException(422, "empty upload")
    return bytes(buf)


def _open_rgb(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")
    except Exception as exc:  # truncated, not an image, decompression bomb…
        raise HTTPException(422, f"cannot decode image: {exc}") from exc


def _analyze(ctx: AppState, image: Image.Image, base_url: str) -> AnalyzeResponse:
    t0 = time.perf_counter()
    with ctx.infer_lock:
        emb = embed_image(ctx.model, ctx.preprocess, image)
        xy = atlas_position(ctx.umap_model, emb)  # numba code is not re-entrant-safe
    embed_ms = (time.perf_counter() - t0) * 1000

    preds = [GenusPrediction(genus=g, subfamily=s, probability=p)
             for g, s, p in predict_genus(ctx.probe, emb, ctx.subfamily_of, k=3)]

    similar = []
    for nb in find_similar(emb, ctx.train_embeddings, k=5):
        code = str(ctx.train_index.at[nb.row, "specimen_code"])
        meta = ctx.specimens.loc[code]
        similar.append(SimilarSpecimen(
            specimen_code=code, genus=str(meta["genus"]), subfamily=str(meta["subfamily"]),
            species=_clean(meta["species"]), similarity=nb.similarity,
            image_url=f"{base_url}images/{code}",
            antweb_url=ANTWEB_SPECIMEN_URL.format(code=code),
            photographer=_clean(meta["creator"]), license=_clean(meta["license"]),
            locality=_clean(meta["locality"])))
    return AnalyzeResponse(predictions=preds, similar=similar,
                           atlas_position=None if xy is None else AtlasPosition(x=xy[0], y=xy[1]),
                           model_name=ctx.model_name, probe_version=ctx.probe_version,
                           embed_ms=round(embed_ms, 1))


@app.post("/analyze", response_model=AnalyzeResponse, responses=_ERROR_RESPONSES)
async def analyze(request: Request, file: UploadFile) -> AnalyzeResponse:
    ctx = _ctx(request)
    max_bytes = int(ctx.config["api_max_upload_mb"]) * (1 << 20)
    data = await _read_upload(file, max_bytes)
    image = _open_rgb(data)
    t0 = time.perf_counter()
    out = await run_in_threadpool(_analyze, ctx, image, str(request.base_url))
    out.filename = file.filename
    log.info("analyze %s %dx%d %.1f KB -> %s (%.2f) atlas %s in %.0f ms",
             file.filename, *image.size, len(data) / 1024, out.predictions[0].genus,
             out.predictions[0].probability,
             "null" if out.atlas_position is None else
             f"({out.atlas_position.x:.2f}, {out.atlas_position.y:.2f})",
             (time.perf_counter() - t0) * 1000)
    return out


# -------------------------------------------------------------------- /genera
@app.get("/genera", response_model=GeneraResponse)
async def genera(request: Request) -> GeneraResponse:
    ctx = _ctx(request)
    n_train = ctx.train_index["genus"].value_counts()
    rows = [GenusInfo(name=g, subfamily=str(r["subfamily"]),
                      n_train=int(n_train.get(g, 0)), n_test=int(r["n_test"]),
                      f1_probe=float(r["f1_linear_probe"]),
                      atlas_median=AtlasPosition(x=ctx.genus_atlas_median[g][0],
                                                 y=ctx.genus_atlas_median[g][1]))
            for g, r in ctx.per_genus.sort_index().iterrows()]
    return GeneraResponse(genera=rows, n=len(rows))


# --------------------------------------------------------------------- /atlas
@app.get("/atlas", response_model=AtlasResponse)
async def atlas(request: Request) -> Response:
    """All 1,236 specimens on the UMAP. The JSON body is rendered once
    (validated through AtlasResponse) and reused for every call."""
    ctx = _ctx(request)
    cached = getattr(request.app.state, "atlas_json", None)
    if cached is None:
        body = AtlasResponse(points=ctx.atlas, n=len(ctx.atlas), umap=ctx.atlas_params)
        cached = request.app.state.atlas_json = body.model_dump_json().encode()
        log.info("atlas payload cached: %d points, %.0f KB", len(ctx.atlas), len(cached) / 1024)
    return Response(content=cached, media_type="application/json",
                    headers={"Cache-Control": "public, max-age=3600"})


# --------------------------------------------------------------- /geo/{genus}
@app.get("/geo/{genus}", response_model=GeoResponse,
         responses={404: {"model": ErrorResponse}})
async def geo(request: Request, genus: str) -> GeoResponse:
    ctx = _ctx(request)
    sub = ctx.full[ctx.full["genus"] == genus]
    if sub.empty:
        raise HTTPException(404, f"unknown genus {genus!r}")
    elev = sub["elevation"].dropna()
    years = sub["year"].dropna()
    coords = sub[["decimalLatitude", "decimalLongitude"]].dropna()
    cap = int(ctx.config["api_geo_max_points"])
    capped = len(coords) > cap
    if capped:
        coords = coords.sample(cap, random_state=int(ctx.config["seed"]))
    summary = ctx.geo_summary.loc[genus] if genus in ctx.geo_summary.index else None
    return GeoResponse(
        genus=genus, subfamily=str(sub["subfamily"].iloc[0]), n_specimens=len(sub),
        n_species=0 if summary is None else int(summary["n_species"]),
        n_unidentified=0 if summary is None else int(summary["n_unidentified"]),
        provinces={str(k): int(v) for k, v in sub["province"].value_counts().items()},
        elevation=ElevationStats(min=float(elev.min()) if len(elev) else None,
                                 median=float(elev.median()) if len(elev) else None,
                                 max=float(elev.max()) if len(elev) else None,
                                 n=int(len(elev))),
        year_min=int(years.min()) if len(years) else None,
        year_max=int(years.max()) if len(years) else None,
        n_with_coords=int(sub[["decimalLatitude", "decimalLongitude"]].notna().all(axis=1).sum()),
        points=[(round(float(a), 5), round(float(b), 5)) for a, b in coords.to_numpy()],
        points_capped=capped)


# ------------------------------------------------------- /images/{specimen_code}
@app.get("/images/{specimen_code}", response_class=FileResponse,
         responses={404: {"model": ErrorResponse}}, name="get_image")
async def get_image(request: Request, specimen_code: str) -> FileResponse:
    ctx = _ctx(request)
    if specimen_code not in ctx.specimens.index:  # also blocks path traversal
        raise HTTPException(404, f"unknown specimen {specimen_code!r}")
    path = ROOT / str(ctx.specimens.at[specimen_code, "image_path"])
    if not path.is_file():
        raise HTTPException(404, f"no image on disk for {specimen_code!r}")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# -------------------------------------------------------------------- /health
@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        return HealthResponse(status="loading", model_loaded=False, n_embeddings=0,
                              n_train=0, n_genera=0, versions={})
    return HealthResponse(status="ok", model_loaded=ctx.model is not None,
                          n_embeddings=int(ctx.embeddings.shape[0]),
                          n_train=int(len(ctx.train_index)),
                          n_genera=int(len(ctx.probe.classes_)), versions=ctx.versions())


# CORS: origins come from config.yaml so the Next.js dev origin isn't hard-coded.
from api.state import load_config  # noqa: E402  (cheap: yaml only)

app.add_middleware(CORSMiddleware,
                   allow_origins=list(load_config(ROOT / "config.yaml")["api_cors_origins"]),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
