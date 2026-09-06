#!/usr/bin/env python3
"""Build reports/rise_onepager.pdf: one A4 page, the four most informative
RISE figures from reports/rise/ with three sentences of context, in the same
style as the deck (11_deck.py). Reads reports/rise/runs.json for the
per-figure probabilities.

    .venv/bin/python scripts/13_rise_onepager.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RISE = REPORTS / "rise"
OUT_PDF = REPORTS / "rise_onepager.pdf"
ACCENT = "#534AB7"

# The four figures shown, in order, with a one-line reading each.
FIGURES: list[tuple[str, str]] = [
    ("casent0104549_Odontomachus.png",
     "<i>Odontomachus</i>: the hotspot sits on the head capsule and the mandible "
     "bases, exactly the region a key uses to separate the trap-jaw genera."),
    ("casent0102933_Anochetus.png",
     "<i>Anochetus</i>, the smaller trap-jaw look-alike: the same head-and-"
     "mandible region carries the decision on this specimen too."),
    ("casent0499794_Technomyrmex.png",
     "<i>Technomyrmex</i>: the gaster is the hottest region - plausibly the "
     "gastral segmentation that separates it from <i>Tapinoma</i>, which on the "
     "same photo is read from the head instead."),
    ("casent0005580_Strumigenys.png",
     "<i>Strumigenys</i>, the probe's best genus, and an honest surprise: the "
     "map favours the head capsule and the gaster, not the famous trap jaws."),
]

CONTEXT = (
    "RISE (Petsiuk et al. 2018, arXiv:1806.07421) covers a photograph with "
    "random coarse masks, scores every masked image with the frozen BioCLIP 2 "
    "tower and the linear probe, and averages the masks weighted by the "
    "probability of a target genus, so the map shows what the deployed "
    "black-box pipeline actually relies on. This reproduces the analysis of "
    "Fisher et al.'s unpublished 2018 study on our model: same question, same "
    "method, new backbone. Maps are qualitative - the blob scale is one cell "
    "of the 7×7 mask grid, and the weights use the probe's raw softmax "
    "because the deployed temperature-calibrated probabilities saturate at "
    "1.0 and carry no signal - but they are how we check that a right answer "
    "is right for a defensible reason."
)


def build_html() -> str:
    runs = json.loads((RISE / "runs.json").read_text())
    n, grid, p = runs["n_masks"], runs["grid"], runs["p_keep"]
    figs = ""
    for fname, caption in FIGURES:
        path = RISE / fname
        if not path.exists():
            raise SystemExit(f"missing figure {path}")
        figs += f"""
      <figure>
        <img src="{path.resolve().as_uri()}">
        <figcaption>{caption}</figcaption>
      </figure>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400;1,8..60,600&display=swap" rel="stylesheet">
<style>
  @page {{ size: A4; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: #fff; color: #141414;
    font-family: Inter, system-ui, sans-serif; font-size: 12px; line-height: 1.45; }}
  .page {{ width: 210mm; height: 297mm; padding: 16mm 16mm 12mm; display: flex; flex-direction: column; }}
  .eyebrow {{ font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: #8a8880;
    font-weight: 500; margin-bottom: 6px; }}
  h1 {{ font-family: "Source Serif 4", Georgia, serif; font-weight: 600; font-size: 26px;
    letter-spacing: -0.01em; margin: 0 0 10px; }}
  .context {{ font-size: 12px; color: #4f4e4a; border-left: 3px solid {ACCENT};
    background: #f8f8f6; padding: 8px 12px; border-radius: 0 4px 4px 0; margin-bottom: 12px; }}
  .grid {{ flex: 1; min-height: 0; display: grid; grid-template-rows: 1fr 1fr 1fr 1fr; gap: 8px; }}
  figure {{ margin: 0; min-height: 0; display: flex; flex-direction: column; align-items: center; }}
  figure img {{ flex: 1; min-height: 0; max-width: 100%; width: auto; object-fit: contain;
    border: 0.5px solid rgba(20,20,20,0.18); border-radius: 4px; background: #fff; }}
  figcaption {{ font-size: 10.5px; color: #6b6a65; margin-top: 2px; align-self: stretch; }}
  i {{ font-style: italic; }}
  .foot {{ margin-top: 8px; display: flex; justify-content: space-between; font-size: 9.5px;
    color: #8a8880; border-top: 0.5px solid rgba(20,20,20,0.18); padding-top: 5px; }}
</style></head><body>
<div class="page">
  <div class="eyebrow">Vitsika · saliency analysis</div>
  <h1>Where the model looks: RISE maps on the deployed pipeline</h1>
  <p class="context">{CONTEXT} Parameters: {n} masks, {grid}×{grid} grid, keep probability {p}.</p>
  <div class="grid">{figs}
  </div>
  <div class="foot">
    <span>Lucas R. · aina@lucas-ralambo.com · vitsika.lucas-ralambo.com/saliency · github.com/LucasRal/vitsika</span>
    <span>Images © their photographers via AntWeb (CC BY-SA)</span>
  </div>
</div>
</body></html>"""


def main() -> None:
    from playwright.sync_api import sync_playwright
    html = build_html()
    tmp = RISE / "_onepager.html"
    tmp.write_text(html)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        page.pdf(path=str(OUT_PDF), width="210mm", height="297mm",
                 print_background=True, page_ranges="1")
        browser.close()
    tmp.unlink()
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
