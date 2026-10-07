# Vitsika (mg-ants)

Genus classifier for Malagasy ants: BioCLIP 2 image embeddings and a
temperature-scaled logistic-regression probe, a FastAPI backend (`api/`) and a
Next.js site (`web/`), live at https://vitsika.lucas-ralambo.com. Proof of
concept for a research paper with the California Academy of Sciences.
Start with README.md, ENVIRONMENT.md, README-web.md and DEPLOY.md.

## Standing rules

- Never modify `data/images`. Write in `data/` only where a request names the output.
- The POC baseline is frozen at tag `poc-baseline`: never change its reports.
  New work goes in new files and folders.
- Commit only when asked. Author: Lucas Ralambo <aina@lucas-ralambo.com>.
  No Co-Authored-By line and no other AI attribution anywhere. One commit per
  logical unit. Never push unless asked.
- No em dashes in any text in the repo.
- Secrets live only in `/etc/vitsika/api.env` on the VPS, never in the repo or in output.
- No network step (downloads, GBIF, Hugging Face) unless a request explicitly
  asks for it. Models are cached locally; set `HF_HUB_OFFLINE=1`.
- Deliverables and contact email: aina@lucas-ralambo.com.

## Conventions

- Numbered scripts `scripts/NN_name.py`, each with a docstring stating what it
  reads and writes.
- Logs via `setup_logging()` from `scripts/gbif_client.py` into
  `reports/<topic>/NN_name.log`.
- Digit-leading scripts are imported with `importlib.util.spec_from_file_location`.
- Fixed seeds (42).
- Reports: lead with the conclusion, short sentences, numbers in tables, file
  and specimen-code references so every claim can be checked.

## Machines and flow

- **This Mac**: development. Same checkout as the VPS; `.venv` and
  `web/node_modules` rebuilt locally.
- **GitHub**: the hub. Remote `origin` = `git@github-lucasral:LucasRal/vitsika.git`.
- **VPS** (ssh alias `ovh-devbox`): production runs straight from
  `/home/ubuntu/projects/lifeplan/mg-ants` (API on 127.0.0.1:8050, site on
  3050, nginx in front). Redeploy = the "Redeploy" block of DEPLOY.md.
- Git carries code and reports only. `data/` is gitignored except `data/*.csv`:
  images, embeddings, probe, UMAP model and `data/segmented` exist only as
  files on each machine.

## Run the app (Mac)

```bash
# API, repo root
HF_HUB_OFFLINE=1 .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8001
# site, http://localhost:3000 (API URL from web/.env.development)
cd web && pnpm dev
# checks
cd web && pnpm test && pnpm lint && pnpm build
```

Never create `web/.env.local` on the VPS (it outranks `.env.production` at build time).
