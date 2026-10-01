"""Scoped chat about one RISE saliency map ("Discuss this map" on /saliency).

  GET  /saliency/chat/status   {"enabled": bool, "model": str | null}
  POST /saliency/chat          {specimen, target, messages[]} -> streamed plain text

The model is any LiteLLM vision model named in LLM_VISION_MODEL (for example
"anthropic/claude-sonnet-4-5" or "openai/gpt-4o") with that provider's key in
the environment; when either is missing the status route reports disabled
and the web UI hides the button. The system prompt is built server-side from
the specimen photo and both overlays (attached as images on the first user
turn), the map's numbers in reports/rise/runs.json, the probe's top-3 for the
cached embedding and the five retrieval receipts, and it confines the model
to describing the evidence of this one map. Every exchange is appended to
reports/saliency_chat_log.jsonl. Rate limit: 10 messages per minute per IP;
history is capped at the last 12 messages.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.inference import find_similar, predict_genus
from api.state import REPORTS, AppState

log = logging.getLogger("api")
router = APIRouter(prefix="/saliency/chat", tags=["saliency"])

RISE_DIR = REPORTS / "rise"
RUNS_JSON = RISE_DIR / "runs.json"
CHAT_LOG = REPORTS / "saliency_chat_log.jsonl"
MODEL_ENV = "LLM_VISION_MODEL"
REASONING_ENV = "LLM_REASONING_EFFORT"   # optional: none (default on reasoning models), low, medium, high
TEMPERATURE = 0.3
MAX_REPLY_TOKENS = 260          # ~120 words with headroom
MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 2000
RATE_LIMIT = 10                 # messages
RATE_WINDOW_S = 60.0
REGIONS = "head, mandibles, antennae, mesosoma, petiole, gaster, legs, background"

_log_lock = threading.Lock()
_rate_lock = threading.Lock()
_recent: dict[str, deque[float]] = defaultdict(deque)
_runs_cache: tuple[float, dict[str, Any]] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    specimen: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    target: str = Field(pattern=r"^[A-Za-z]{1,40}$")
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)


class ChatStatus(BaseModel):
    enabled: bool
    model: str | None = None


# --------------------------------------------------------------- availability
def _model_name() -> str | None:
    return os.environ.get(MODEL_ENV) or None


def chat_enabled() -> tuple[bool, str | None]:
    """Enabled only when LLM_VISION_MODEL is set and LiteLLM finds the
    provider key for it in the environment."""
    model = _model_name()
    if not model:
        return False, None
    try:
        import litellm
        ok = bool(litellm.validate_environment(model).get("keys_in_environment"))
    except Exception as exc:  # noqa: BLE001 (litellm raises assorted types on odd names)
        log.warning("saliency chat: cannot validate %s=%r: %s", MODEL_ENV, model, exc)
        ok = False
    return ok, model


@router.get("/status", response_model=ChatStatus)
async def status() -> ChatStatus:
    ok, model = chat_enabled()
    return ChatStatus(enabled=ok, model=model if ok else None)


# ------------------------------------------------------------------- context
def _runs() -> dict[str, Any]:
    """reports/rise/runs.json, re-read when its mtime changes."""
    global _runs_cache
    mtime = RUNS_JSON.stat().st_mtime
    if _runs_cache is None or _runs_cache[0] != mtime:
        _runs_cache = (mtime, json.loads(RUNS_JSON.read_text()))
    return _runs_cache[1]


def _run_for(specimen: str, target: str) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = _runs()
    for r in meta["runs"]:
        if r["specimen_code"] == specimen and r["target"] == target:
            return meta, r
    raise HTTPException(404, f"no saliency map for {specimen!r} / {target!r}")


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise HTTPException(404, f"missing figure {path.name}")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _evidence(ctx: AppState, specimen: str) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Top-3 probe scores and five retrieval receipts for the specimen's
    cached embedding (same probe and neighbour search as /analyze)."""
    rows = ctx.index.index[ctx.index["specimen_code"] == specimen]
    if len(rows) == 0:
        raise HTTPException(404, f"no embedding for {specimen!r}")
    emb = ctx.embeddings[int(rows[0])]
    top3 = predict_genus(ctx.probe, emb, ctx.subfamily_of, k=3)
    receipts = [(str(ctx.train_index.at[nb.row, "specimen_code"]),
                 str(ctx.train_index.at[nb.row, "genus"]), nb.similarity)
                for nb in find_similar(emb, ctx.train_embeddings, k=5)]
    return top3, receipts


def build_system_prompt(ctx: AppState, meta: dict[str, Any], run: dict[str, Any]) -> str:
    specimen, target = run["specimen_code"], run["target"]
    top3, receipts = _evidence(ctx, specimen)
    species = run.get("species") or "species not recorded"
    top3_txt = ", ".join(f"{g} {p:.3f}" for g, _, p in top3)
    receipts_txt = "; ".join(f"{code} ({genus}, cosine {s:.3f})" for code, genus, s in receipts)
    return (
        "You are discussing ONE saliency map from the Vitsika ant-genus classifier "
        "(BioCLIP 2 embeddings + a linear probe), computed with RISE: random masks hide "
        "parts of the photograph, the classifier is scored each time, and the map averages "
        "the masks weighted by the target-genus probability.\n\n"
        f"Specimen: {specimen}, labelled genus {run['true_genus']} ({species}).\n"
        f"Target genus of this map: {target}.\n"
        f"Numbers for this map (RISE weights use the probe's raw softmax, T = 1, because the "
        f"deployed calibrated probabilities saturate at 1.0 under masking): full-image "
        f"P({target}) = {run['p_full']:.3f}; "
        f"mean P over the {meta['n_masks']} masked images = {run['p_masked_mean']:.3f}, "
        f"max = {run['p_masked_max']:.3f}; map baseline (mean saliency) = {run.get('baseline', run['p_masked_mean']):.3f}; "
        f"saliency range {run.get('sal_min', 0):.3f} to {run.get('sal_max', 0):.3f}; "
        f"masks: {meta['n_masks']}, {meta['grid']}x{meta['grid']} grid, keep probability {meta['p_keep']}, "
        f"seed {meta['seed']}; shared deviation scale half-width W = {meta.get('shared_w', 0):.3f} "
        "(the largest deviation on the whole saliency page).\n"
        f"Deployed (temperature-calibrated) probe top-3 for this photograph, as the site "
        f"shows it: {top3_txt}.\n"
        f"Five retrieval receipts (nearest training specimens): {receipts_txt}.\n\n"
        "Images attached on the first turn, in order: (1) the photograph, (2) the per-map "
        "overlay, stretched to this map's own range, (3) the shared-scale overlay, deviation "
        "from the map's baseline on the page-wide scale.\n\n"
        "Rules:\n"
        "- Answer what was actually asked. A greeting, thanks or small talk gets a short, "
        "courteous reply in one or two sentences and an offer to discuss the map; do not "
        "launch into an analysis nobody asked for.\n"
        "- Red means that seeing that part of the photo raises the target score; blue means "
        "it lowers it; the middle colour means no effect. On the per-map overlay the colours "
        "are stretched to this map's own range, so always check the numbers for magnitude.\n"
        f"- Name body regions only from this list: {REGIONS}. Regions are read from the "
        "overlay by eye, not measured.\n"
        "- Never assert a taxonomic character (no claims about what a structure diagnoses) "
        "and never make a species-level claim.\n"
        "- If asked something the map cannot answer (a new species, field identification, "
        "other specimens, whether the label is right), say plainly what this map cannot do "
        "and offer what it can: where the evidence for the target genus sits on this one photo.\n"
        "- At most 120 words per reply, plain English, no bullet lists unless asked."
    )


def build_messages(ctx: AppState, req: ChatRequest) -> list[dict[str, Any]]:
    meta, run = _run_for(req.specimen, req.target)
    history = req.messages[-MAX_HISTORY:]
    while history and history[0].role != "user":   # providers require a user turn first
        history = history[1:]
    if not history or history[-1].role != "user":
        raise HTTPException(422, "the last message must be from the user")
    images = [_data_url(RISE_DIR / "web" / f"{req.specimen}_photo.png"),
              _data_url(RISE_DIR / "web" / f"{req.specimen}_{req.target}_overlay.png"),
              _data_url(RISE_DIR / "web" / f"{req.specimen}_{req.target}_overlay_shared.png")]
    out: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(ctx, meta, run)}]
    first_user_done = False
    for m in history:
        if m.role == "user" and not first_user_done:
            first_user_done = True
            out.append({"role": "user", "content": [
                {"type": "text", "text": m.content},
                *({"type": "image_url", "image_url": {"url": u}} for u in images)]})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


# ---------------------------------------------------------------- rate limit
def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate(ip: str) -> None:
    now = time.monotonic()
    with _rate_lock:
        q = _recent[ip]
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, f"rate limit: {RATE_LIMIT} messages per minute; try again shortly")
        q.append(now)


def _append_log(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False)
    with _log_lock:
        with CHAT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def sampling_params(model: str) -> dict[str, Any]:
    """temperature + reasoning_effort for this model.

    Reasoning models (OpenAI gpt-5.x, Claude with extended thinking) default to
    reasoning_effort "none": their thinking tokens would otherwise eat the
    reply budget, and gpt-5.x rejects any temperature but 1 while reasoning is
    on. LLM_REASONING_EFFORT overrides; a non-"none" value drops temperature.
    Providers without the notion ignore the parameter (LiteLLM strips it).
    """
    import litellm
    effort = os.environ.get(REASONING_ENV, "").strip().lower() or None
    if effort is None:
        try:
            if litellm.supports_reasoning(model=model):
                effort = "none"
        except Exception:  # noqa: BLE001 (unknown model name: leave the default alone)
            pass
    params: dict[str, Any] = {}
    if effort is not None:
        params["reasoning_effort"] = effort
    if effort in (None, "none"):
        params["temperature"] = TEMPERATURE
    return params


# --------------------------------------------------------------------- route
@router.post("")
async def chat(request: Request, req: ChatRequest) -> StreamingResponse:
    enabled, model = chat_enabled()
    if not enabled:
        raise HTTPException(503, "saliency chat is not configured on this server")
    _check_rate(_client_ip(request))
    ctx: AppState = request.app.state.ctx
    messages = build_messages(ctx, req)
    history = [m.model_dump() for m in req.messages[-MAX_HISTORY:]]

    import litellm
    litellm.suppress_debug_info = True
    try:
        stream = await litellm.acompletion(model=model, messages=messages, stream=True,
                                           max_tokens=MAX_REPLY_TOKENS, **sampling_params(model))
    except Exception as exc:  # noqa: BLE001 (provider errors are many-typed)
        log.warning("saliency chat: %s failed: %s", model, exc)
        raise HTTPException(502, f"the language model did not answer ({type(exc).__name__})") from exc

    t0 = time.perf_counter()

    async def body() -> AsyncIterator[bytes]:
        parts: list[str] = []
        error: str | None = None
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield delta.encode()
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            log.warning("saliency chat: stream from %s broke: %s", model, error)
            yield b"\n[the reply was cut short]"
        finally:
            reply = "".join(parts)
            _append_log({"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                         "specimen": req.specimen, "target": req.target, "model": model,
                         "messages": history + [{"role": "assistant", "content": reply}],
                         "error": error, "seconds": round(time.perf_counter() - t0, 2)})
            log.info("saliency chat %s/%s: %d msgs in, %d chars out, %.1fs%s", req.specimen,
                     req.target, len(history), len(reply), time.perf_counter() - t0,
                     f" ({error})" if error else "")

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
