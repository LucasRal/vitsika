# Vitsika: web front end (`web/`)

Next.js 15 (App Router, TypeScript, `src/` layout), Tailwind CSS v4,
ECharts (`echarts-for-react`) for the atlas, `react-leaflet` 5 with
OpenStreetMap tiles for the distribution maps. No component library: every
control is hand-rolled in `src/components/ui.tsx`. Fonts via `next/font`:
Source Serif 4 (display), Inter (UI), JetBrains Mono (specimen codes).

## Development

```bash
# 1. the API (repo root)
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8001

# 2. the site
cd web
# dev API URL comes from .env.development; create .env.local only to override it
pnpm install
pnpm dev                              # http://localhost:3000
pnpm lint && pnpm build               # type-check + production build
```

`pnpm` is not installed system-wide on the dev box; `npm i -g pnpm` with
`npm config set prefix ~/.npm-global` (and that on `PATH`) does it.

### Environment variables

| Variable | Where | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | browser + server | Base URL of the FastAPI backend as the **browser** sees it. Dev: `http://localhost:8001` (`.env.development`). Behind nginx: `/api` (`.env.production`). Inlined at build time, and `.env.local` outranks `.env.production` even for `next build`, so never leave a dev `.env.local` on the server. |
| `API_URL` | server only, optional | Absolute base used by server components when the public one is relative (`http://127.0.0.1:8001`). |

CORS: the API only allows the origins in `config.yaml` → `api_cors_origins`
(`http://localhost:3000` by default). Behind a same-origin reverse proxy CORS
does not apply.

## Pages

| Route | Type | Data |
|---|---|---|
| `/` Identify | server page, client dropzone | `GET /genera` (model card) + static `src/data/metrics.json` (copy of `reports/metrics.json`); `POST /analyze` on submit |
| `/result` | client (state from the Identify page; nothing persisted) | thumbnails from `GET /images/{code}` |
| `/genera` | server fetch, client table (sort / search / filter) | `GET /genera` |
| `/distribution` | server fetch of genera, client map | `GET /geo/{genus}`; grid clustering above 200 points, no plugin |
| `/atlas` | server fetch, client ECharts | `GET /atlas`; `?star=1&x=&y=&label=` renders the upload as a star |
| `/methods` | static | - |

The two atlas annotations are hard-coded medians from `data/umap_coords.csv`
(long-legged island 3.51, 9.23; *Anochetus* island 5.48, 9.53). Re-fit the
UMAP → update them in `src/app/atlas/AtlasChart.tsx`.

## Production

```bash
cd web && pnpm install --frozen-lockfile && pnpm build   # reads web/.env.production
pnpm start -p 3050                                        # API_URL=http://127.0.0.1:8050 from .env.production
```

Production ports are **3050** (Next) and **8050** (API); dev keeps 3000/3001 and
8001. The systemd units and nginx site live in `deploy/`; see `DEPLOY.md`.

`NEXT_PUBLIC_*` is inlined at build time: set it before `pnpm build`. The
data pages are `force-dynamic` (rendered per request, API responses cached
5 min / 1 h via `fetch` revalidation), so the build does not need the API
up and a down API only ever shows the banner for the current request.

Keep both processes alive with pm2 …

```bash
pm2 start ".venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8050 --root-path /api" --name vitsika-api --cwd /srv/mg-ants
pm2 start "pnpm start -p 3050" --name vitsika-web --cwd /srv/mg-ants/web
pm2 save && pm2 startup
```

… or systemd (one unit each, `WorkingDirectory=` as above,
`Environment=NEXT_PUBLIC_API_URL=/api`). `--root-path /api` makes the API's
own absolute URLs (`image_url` in `/analyze`, the OpenAPI docs) include the
prefix; the front end builds image URLs itself from `specimen_code`, so it
works either way.

nginx, same origin for site and API:

```nginx
server {
    server_name vitsika.example.org;
    client_max_body_size 12m;                      # the API rejects > 10 MB itself

    location /api/ {
        proxy_pass http://127.0.0.1:8050/;         # trailing slash strips /api
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;                    # ~1 s per /analyze on CPU
    }
    location / {
        proxy_pass http://127.0.0.1:3050;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;    # Next.js HMR only matters in dev
        proxy_set_header Connection "upgrade";
    }
}
```

Startup of the API takes ~30 s (BioCLIP 2 weights + UMAP JIT); `/health`
returns `{"status":"loading"}` until then and every page shows a
"service not reachable" banner instead of crashing.

## Notes

- **react-leaflet 5, not 4.** Next 15 ships React 19; react-leaflet 4.2.1
  declares `react ^18` and its `MapContainer` throws
  `Map container is already initialized` under React 19's StrictMode ref
  semantics (the whole page tree unmounts in dev). v5 has the same API and
  targets React 19, so the map code is unchanged.
- The dev box already had servers on :3000 and :8000, so this site is
  exercised on :3001 against the API on :8001; `config.yaml` allows both
  `localhost:3000` and `localhost:3001` as CORS origins.
- `notebooks/exercise_web.py` (untracked) drives every page in headless
  Chromium via Playwright (upload flow, atlas modes, table sort/filter,
  map clustering, dark mode) and reports console errors;
  `notebooks/ux_tiers.py` covers the verdict tiers, the atlas upload flow
  and the staged progress panel (desktop, mobile, reduced motion, error+retry).
- **Unit tests**: `pnpm test` (vitest). `src/lib/verdict.ts` decides the
  confidence verdict shown on `/result`: *confident* (top-1 ≥ 0.5, no
  banner), *supported* (top-1 < 0.5 but ≥ 3 of the 5 nearest reference
  specimens share the genus at cosine ≥ 0.85 → neutral note) or *low*
  (amber banner). Fixtures in `src/lib/__fixtures__/` are real
  `/api/analyze` payloads from production (Royidris casent0002219,
  Tetraponera casent0012838).
- The last analysis lives in `AnalysisContext` and is mirrored to
  `sessionStorage` (`vitsika.analysis.v1`, image as a data URL when ≤ 2 MB),
  so it survives tab-surfing and a reload; Identify shows it as "Last
  result" with the photo in the box. It is cleared only by
  "← Identify another specimen" (`/?new=1` or the in-page button) or by
  choosing a new photo.
- The Atlas card's "Place your photo on the map" reuses the Identify
  `Dropzone` (`compact` prop, `onDone` callback): one `/analyze` call gives
  the star position and the full answer, which lands in the same
  `AnalysisContext` so "Full analysis →" opens `/result`.
