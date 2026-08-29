# Deploying Vitsika (mg-ants) on the VPS

Production host: **https://vitsika.lucas-ralambo.com** (VPS `137.74.163.49`,
Ubuntu 26.04, user `ubuntu`). Everything runs from this checkout at
`/home/ubuntu/projects/lifeplan/mg-ants`.

## Architecture

```
browser ──HTTPS──▶ nginx (:443, Let's Encrypt; :80 → 301)
                     ├── /api/*  ──▶ 127.0.0.1:8050  vitsika-api.service   uvicorn api.main:app --root-path /api  (1 worker)
                     ├── /*      ──▶ 127.0.0.1:3050  vitsika-web.service   node …/next start                    (Next.js 15)
                     └── /robots.txt  → "Disallow: /"  (served by nginx)
```

- The API is one uvicorn worker on purpose: it holds the 1.7 GB BioCLIP 2
  weights + UMAP reducer (≈2.2–2.5 GB RSS) and serialises inference behind a
  lock; more workers = more RAM, no more throughput on this 6-core CPU.
- `--root-path /api` + nginx's trailing-slash `proxy_pass` strip and re-add
  the prefix, so `/api/health` reaches the app as `/health` and the URLs the
  API generates (`image_url`, OpenAPI) come back as `https://…/api/…`.
- The site is built with `NEXT_PUBLIC_API_URL=/api` (same origin, no CORS
  in play) and reads `API_URL=http://127.0.0.1:8050` at runtime for
  server-side fetches. Both come from `web/.env.production`.
- Ports 3000/8000 belong to other projects on this box and are not used.
- Ports 3050/8050 bind 127.0.0.1 only; ufw allows OpenSSH + `Nginx Full`.

## Files

| What | Live location | Template in repo |
|---|---|---|
| API unit | `/etc/systemd/system/vitsika-api.service` | `deploy/vitsika-api.service` |
| Web unit | `/etc/systemd/system/vitsika-web.service` | `deploy/vitsika-web.service` |
| nginx site | `/etc/nginx/sites-available/vitsika.conf` → `sites-enabled/` | `deploy/vitsika.conf` (pre-certbot; certbot appended the 443 block + redirect in place) |
| TLS | `/etc/letsencrypt/live/vitsika.lucas-ralambo.com/` — renewed by `certbot.timer` (twice daily; `certbot renew --dry-run` to test) | — |
| Env | `web/.env.production` (committed), `web/.env.development` (dev) | — |
| Logs | `journalctl -u vitsika-api`, `journalctl -u vitsika-web`, `/var/log/nginx/vitsika.{access,error}.log`; the API also writes `reports/api.log` | — |

Changed a template? Re-install it and reload:

```bash
sudo install -m 644 deploy/vitsika-api.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart vitsika-api
sudo install -m 644 deploy/vitsika.conf /etc/nginx/sites-available/vitsika.conf && sudo nginx -t && sudo systemctl reload nginx
# NB: the live vitsika.conf also carries certbot's 443 block — after
# re-installing the template run `sudo certbot --nginx -d vitsika.lucas-ralambo.com --reinstall --redirect`
# (or re-apply the block by hand) so HTTPS comes back.
```

## Redeploy

```bash
cd /home/ubuntu/projects/lifeplan/mg-ants
git pull

# API changed (api/, scripts/, config.yaml, requirements.txt)?
.venv/bin/pip install -r requirements.txt          # only if requirements changed
sudo systemctl restart vitsika-api                  # ~35 s to load the model; /api/health says "loading" meanwhile
journalctl -u vitsika-api -n 20 --no-pager

# Data artefacts changed (embeddings / probe / UMAP)? Re-run the pipeline steps first:
#   scripts/06_embed.py → 07_eval.py → 08_umap.py, then restart the API.
#   The atlas annotations in web/src/app/atlas/AtlasChart.tsx are hard-coded UMAP
#   medians — refresh them if 08_umap.py is re-fitted.

# Site changed (web/)?
cd web
export PATH="$HOME/.npm-global/bin:$PATH"          # pnpm lives there
pnpm install --frozen-lockfile                      # only if package.json changed
pnpm build                                          # ~45 s; reads web/.env.production
sudo systemctl restart vitsika-web                  # ready in ~2 s
```

`next build` writes into `web/.next/` while the old server keeps serving
from it; the restart picks up the new build. For zero-downtime, build into a
copy and swap — not needed at this scale.

Smoke after a redeploy:

```bash
curl -s https://vitsika.lucas-ralambo.com/api/health | head -c 200
curl -s -F "file=@data/images/Royidris/casent0002219_p.jpg" https://vitsika.lucas-ralambo.com/api/analyze | head -c 300
.venv/bin/python scripts/10_smoke_api.py --base-url https://vitsika.lucas-ralambo.com/api
```

## Reading logs & health

```bash
systemctl status vitsika-api vitsika-web nginx
journalctl -u vitsika-api -f            # each /analyze logs size, top-1, atlas position, ms
journalctl -u vitsika-web -f
sudo tail -f /var/log/nginx/vitsika.access.log
curl -s https://vitsika.lucas-ralambo.com/api/health   # status, n_embeddings, model/probe/UMAP versions
```

`MemoryMax=6G` on the API unit: if it is ever OOM-killed, `systemctl status`
shows it and `Restart=on-failure` brings it back after 5 s.

## Environment gotcha (Next.js)

Next loads `.env.local` **above** `.env.production` even during `next build`,
so a stray `web/.env.local` with a dev API URL gets inlined into the
production bundle (this bit us once: the site called `localhost:8001`). The
dev value therefore lives in `.env.development`; never create `.env.local`
on the server.
