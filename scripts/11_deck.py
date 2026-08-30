#!/usr/bin/env python3
"""Build reports/vitsika_deck.pdf (6 slides, 16:9) from the repo's own numbers.

Slides are plain HTML + CSS rendered by headless Chromium (Playwright), one
page per slide, then rasterised to PNG previews with pdftoppm. Numbers are
read from reports/metrics.json and reports/dataset_stats.md at build time so
the deck cannot drift from the evaluation; the rest is written from
reports/onepager.md.

    .venv/bin/python scripts/11_deck.py            # -> reports/vitsika_deck.pdf + reports/deck_previews/slide-N.png
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT_PDF = REPORTS / "vitsika_deck.pdf"
PREVIEWS = REPORTS / "deck_previews"
W, H = 1280, 720  # CSS px; PDF page = 13.333 x 7.5 in
ACCENT = "#534AB7"


def stat(md: str, pattern: str) -> str:
    m = re.search(pattern, md)
    if not m:
        raise SystemExit(f"pattern not found in dataset_stats.md: {pattern}")
    return m.group(1)


def build_html() -> str:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    ds = (REPORTS / "dataset_stats.md").read_text()
    m = metrics["methods"]
    n_test, n_train, n_genera = metrics["n_test"], metrics["n_train"], metrics["n_genera"]
    manifest = stat(ds, r"Target manifest \(data/dataset_full.csv\): \*\*(\d+)\*\* images")
    manifest_genera = stat(ds, r"images, \*\*(\d+)\*\* genera")
    obtained = stat(ds, r"Obtained on disk: \*\*(\d+)\*\* \(([\d.]+)%\)")
    obtained_pct = re.search(r"Obtained on disk: \*\*\d+\*\* \(([\d.]+)%\)", ds).group(1)
    commons = stat(ds, r"- commons: (\d+)")
    cache = stat(ds, r"- gbif_cache: (\d+)")
    images = stat(ds, r"Images \(one per specimen\): \*\*(\d+)\*\*")
    today = dt.date.today().isoformat()
    k = lambda v: f"{int(v):,}"  # thousands separators
    fmt = lambda k, f: f"{m[k][f] * 100:.1f} %" if f != "macro_f1" else f"{m[k][f]:.3f}"

    def row(name: str, key: str, strong: bool = False) -> str:
        b = "font-weight:600;color:#141414" if strong else ""
        return (f"<tr style='{b}'><td>{name}</td><td class='num'>{fmt(key, 'top1')}</td>"
                f"<td class='num'>{fmt(key, 'top3')}</td><td class='num'>{fmt(key, 'macro_f1')}</td></tr>")

    img = lambda rel: (ROOT / rel).resolve().as_uri()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400;1,8..60,600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
  @page {{ size: {W}px {H}px; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: #fff; color: #141414; font-family: Inter, system-ui, sans-serif; font-size: 18px; line-height: 1.45; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .slide {{ width: {W}px; height: {H}px; padding: 56px 72px 44px; position: relative; overflow: hidden; page-break-after: always; break-after: page; display: flex; flex-direction: column; }}
  .slide:last-child {{ page-break-after: auto; break-after: auto; }}
  h1, h2 {{ font-family: "Source Serif 4", Georgia, serif; font-weight: 600; letter-spacing: -0.01em; margin: 0; }}
  h1 {{ font-size: 64px; line-height: 1.05; }}
  h2 {{ font-size: 36px; line-height: 1.15; margin-bottom: 22px; }}
  i, .genus {{ font-style: italic; }}
  .eyebrow {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: #8a8880; font-weight: 500; margin-bottom: 14px; }}
  .accent {{ color: {ACCENT}; }}
  .muted {{ color: #6b6a65; }}
  .code {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.86em; }}
  .foot {{ position: absolute; left: 72px; right: 72px; bottom: 26px; display: flex; justify-content: space-between; font-size: 12px; color: #8a8880; border-top: 0.5px solid rgba(20,20,20,0.18); padding-top: 8px; }}
  .wordmark {{ font-family: "Source Serif 4", Georgia, serif; font-weight: 600; }}
  ul.big {{ list-style: none; padding: 0; margin: 0; font-size: 22px; }}
  ul.big li {{ padding: 14px 0 14px 34px; position: relative; border-top: 0.5px solid rgba(20,20,20,0.18); }}
  ul.big li:first-child {{ border-top: 0; }}
  ul.big li::before {{ content: ""; position: absolute; left: 0; top: 26px; width: 14px; height: 14px; border-radius: 50%; background: {ACCENT}; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; flex: 1; min-height: 0; }}
  .frame {{ border: 0.5px solid rgba(20,20,20,0.18); border-radius: 6px; overflow: hidden; background: #f8f8f6; }}
  .frame img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 17px; }}
  th {{ text-align: left; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #8a8880; font-weight: 500; padding: 6px 10px; border-bottom: 0.5px solid rgba(20,20,20,0.18); }}
  td {{ padding: 10px 10px; border-bottom: 0.5px solid rgba(20,20,20,0.12); color: #4f4e4a; }}
  td.num, th.num {{ text-align: right; font-family: "JetBrains Mono", monospace; font-size: 16px; white-space: nowrap; }}
  .chain {{ display: flex; align-items: stretch; gap: 0; margin-top: 8px; }}
  .node {{ flex: 1; border: 0.5px solid rgba(20,20,20,0.18); border-radius: 6px; padding: 18px 18px; background: #fff; min-width: 0; }}
  .node .n {{ font-family: "Source Serif 4", Georgia, serif; font-size: 38px; font-weight: 600; line-height: 1.1; }}
  .node .l {{ font-size: 14.5px; color: #6b6a65; margin-top: 8px; line-height: 1.35; }}
  .node.hi {{ border-color: {ACCENT}; background: rgba(83,74,183,0.06); }}
  .node.hi .n {{ color: {ACCENT}; }}
  .arrow {{ width: 26px; display: flex; align-items: center; justify-content: center; color: {ACCENT}; font-size: 22px; flex: none; }}
  .callouts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-top: 40px; }}
  .callout {{ border-left: 3px solid {ACCENT}; background: #f8f8f6; padding: 22px 24px; font-size: 18px; line-height: 1.45; border-radius: 0 6px 6px 0; }}
  .callout b {{ display: block; font-weight: 600; margin-bottom: 8px; color: #141414; font-size: 20px; }}
  .kicker {{ font-size: 20px; color: #4f4e4a; max-width: 900px; }}
  .takeaway {{ margin-top: 14px; font-size: 20px; border-left: 3px solid {ACCENT}; padding: 8px 16px; background: #f8f8f6; border-radius: 0 6px 6px 0; }}
  ol.steps {{ list-style: none; padding: 0; margin: 0; counter-reset: s; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 26px; }}
  ol.steps li {{ counter-increment: s; border: 0.5px solid rgba(20,20,20,0.18); border-radius: 6px; padding: 22px 22px 18px; font-size: 16.5px; color: #4f4e4a; line-height: 1.42; }}
  ol.steps li::before {{ content: counter(s); display: block; font-family: "Source Serif 4", Georgia, serif; font-size: 40px; color: {ACCENT}; font-weight: 600; line-height: 1; margin-bottom: 10px; }}
  ol.steps li b {{ display: block; color: #141414; font-weight: 600; font-size: 19px; margin-bottom: 8px; }}
</style></head><body>

<!-- S1 -->
<section class="slide">
  <div class="two" style="grid-template-columns: 400px 1fr; gap: 36px; align-items: center;">
    <div>
      <div class="eyebrow">Proof of concept</div>
      <h1>Vitsika</h1>
      <p style="font-family:'Source Serif 4',Georgia,serif;font-size:26px;margin:14px 0 0;line-height:1.25">Malagasy ant identification from a single photograph</p>
      <p class="muted" style="margin:18px 0 0;font-size:15px"><i>vitsika</i> is Malagasy for ant. Genus-level, 27 genera, built on public data and a frozen public model.</p>
      <p style="margin:30px 0 0;font-size:18px"><span class="accent code">vitsika.lucas-ralambo.com</span></p>
      <p style="margin:16px 0 0;font-size:16px">Lucas R. <span class="muted">· Data Scientist · Antananarivo</span><br><span class="muted">{today}</span></p>
    </div>
    <div class="frame" style="height:536px;background:#fff;box-shadow:0 20px 50px rgba(20,20,20,0.10)"><img src="{img('reports/web_screenshots/light-2f-result-current.png')}" style="object-fit:contain"></div>
  </div>
  <div class="foot"><span><span class="wordmark">Vitsika</span> · the result screen for the example specimen · images © their photographers via AntWeb (CC BY-SA)</span><span>1 / 6</span></div>
</section>

<!-- S2 -->
<section class="slide">
  <div class="eyebrow">Problem</div>
  <h2>Identification is the bottleneck, and it happens far from the field</h2>
  <div class="two" style="grid-template-columns: 1.15fr 1fr; align-items:start">
    <ul class="big">
      <li>AntWeb holds <b>141,430</b> Malagasy ant records, <b>6,995</b> of them with images, across <b>60</b> imaged genera. Naming even the genus takes a specialist and a microscope.</li>
      <li>The specimens, the imaging and the determinations are curated and published by the California Academy of Sciences (San Francisco); the expertise sits outside Madagascar, and a field team waits for it.</li>
      <li>A photo-to-genus first pass that runs on a CPU box in Antananarivo in <b>~0.8 s</b> per image lets a local team triage, then send only the hard cases to a specialist.</li>
    </ul>
    <div>
      <div class="frame" style="height:420px"><img src="{img('reports/contact_sheet.jpg')}"></div>
      <p class="muted" style="font-size:12.5px;margin:8px 0 0">One AntWeb profile shot per genus in scope (27). Sources: reports/01_explore.txt, README.md.</p>
    </div>
  </div>
  <div class="foot"><span>reports/01_explore.txt · README.md § Serving</span><span>2 / 6</span></div>
</section>

<!-- S3 -->
<section class="slide">
  <div class="eyebrow">Data</div>
  <h2>From GBIF to {k(images)} clean images in {n_genera} genera</h2>
  <div class="chain">
    <div class="node"><div class="n">141,430</div><div class="l">AntWeb records for Madagascar on GBIF (doi:10.15468/wqmjjt); 6,995 with an image</div></div>
    <div class="arrow">→</div>
    <div class="node"><div class="n">{k(manifest)}</div><div class="l">specimens, {manifest_genera} genera: workers, profile view, one image each, ≥ 10 per genus</div></div>
    <div class="arrow">→</div>
    <div class="node"><div class="n">{k(obtained)}</div><div class="l">images obtained ({obtained_pct} %): {k(commons)} Wikimedia Commons + {cache} GBIF cache</div></div>
    <div class="arrow">→</div>
    <div class="node hi"><div class="n">{k(images)}</div><div class="l">images, <b>{n_genera} genera</b>, 8 subfamilies; {n_train} train / {n_test} held-out test</div></div>
  </div>
  <div class="callouts">
    <div class="callout"><b>Finding 1: GBIF's image cache is empty for AntWeb</b>
      4,215 of 4,354 cache requests returned 404 (96.8 %), uniformly across years and record age. antweb.org blocks scripts (Cloudflare) and datacenter IPs (403), including GBIF's own fetcher. The working source is the 2008-2014 bulk upload to Wikimedia Commons (33,081 files), matched by specimen code.</div>
    <div class="callout"><b>Finding 2: caste is not in GBIF's interpreted fields</b>
      GBIF maps <span class="code">sex</span> to Male / Female / Other, so every worker arrives as null: 4,540 of 5,857 specimens. The caste has to be read from the verbatim record (<span class="code">dwc:sex</span>): 4,539 workers, 673 queens, 640 males, and only workers are kept.</div>
  </div>
  <div class="foot"><span>reports/dataset_stats.md · reports/download_stats.md · reports/01_explore.txt · data/raw/records.parquet</span><span>3 / 6</span></div>
</section>

<!-- S4 -->
<section class="slide">
  <div class="eyebrow">Results</div>
  <h2 style="font-size:32px">Linear probe on frozen BioCLIP 2 embeddings: {fmt('linear_probe','top1')} top-1 on {n_test} held-out specimens</h2>
  <div class="two" style="grid-template-columns: 1fr 1.1fr; align-items:start">
    <div>
      <table>
        <thead><tr><th>Method</th><th class="num">Top-1</th><th class="num">Top-3</th><th class="num">Macro-F1</th></tr></thead>
        <tbody>
          {row("Zero-shot, “a photo of {genus}, a genus of ant”", "zero_shot_template")}
          {row("Linear probe (deployed)", "linear_probe", True)}
          {row("Nearest neighbour, cosine", "nearest_neighbour")}
        </tbody>
      </table>
      <p class="muted" style="font-size:13px;margin:10px 0 0">{n_train} train / {n_test} test, {n_genera} genera, split by genus (seed 42). Bare genus name as the zero-shot prompt: {fmt('zero_shot_plain','top1')} top-1. Embeddings: <span class="code">imageomics/bioclip-2</span> ViT-L/14, 768-d, no fine-tuning.</p>
      <div class="takeaway" style="font-size:17px">18 errors: 12 genuine look-alikes or species unseen in training (<i>Syllophopsis</i> vs <i>Tetramorium</i>, <i>Royidris</i> vs <i>Monomorium</i>), 3 poor photos, 2 odd angles, 1 unexplained. Almost every confusion stays inside a subfamily block.</div>
    </div>
    <div class="frame" style="height:500px;background:#fff"><img src="{img('reports/confusion_matrix.png')}"></div>
  </div>
  <div class="foot"><span>reports/metrics.json · reports/confusion_matrix.png · reports/eval_notes.md</span><span>4 / 6</span></div>
</section>

<!-- S5 -->
<section class="slide">
  <div class="eyebrow">Structure</div>
  <h2>The embedding space recapitulates taxonomy, with body-plan islands</h2>
  <div class="frame" style="flex:1;min-height:0;background:#fff"><img src="{img('reports/umap_by_subfamily_annotated.png')}"></div>
  <div class="takeaway">Subfamilies form clean clusters; where the model "errs", it groups <i>Camponotus imitator</i> with long-legged <i>Aphaenogaster</i> and <i>Odontomachus</i>, a resemblance a taxonomist would recognise, not noise.</div>
  <div class="foot"><span>reports/umap_by_subfamily_annotated.png (data/umap_coords.csv) · reports/eval_notes.md § Addendum</span><span>5 / 6</span></div>
</section>

<!-- S6 -->
<section class="slide">
  <div class="eyebrow">Next steps</div>
  <h2>Three things this codebase already supports</h2>
  <ol class="steps">
    <li><b>Complete the image set</b> 70.2 % of the manifest's images are missing. <span class="code">05_download_antweb.py</span> + <span class="code">dataset_full.csv</span> are ready for a residential IP or an AntWeb bulk export; re-thresholding brings <i>Vitsika</i>, <i>Tanipone</i>, <i>Carebara</i> and 9 more genera into scope (39 in total).</li>
    <li><b>Honest generalisation and an open set</b> Group the split by species (one flag in <span class="code">03_build_dataset.py</span>), calibrate the probe's probabilities on the held-out set, and turn the nearest-neighbour similarity the API already returns into a "none of the 27" reject.</li>
    <li><b>Multi-view, multi-caste embeddings</b> The harvest already holds 6,634 head and 5,127 dorsal media rows plus 673 queens and 640 males; <span class="code">config.yaml</span> (<span class="code">views_priority</span>, <span class="code">castes</span>) and <span class="code">06_embed.py</span> make view-wise embedding and late fusion a configuration change.</li>
  </ol>
  <p style="margin-top:auto;font-size:18px">Lucas R. · Data Scientist, Antananarivo · <span class="accent code">aina@lucas-ralambo.com</span> · <span class="accent code">vitsika.lucas-ralambo.com</span></p>
  <div class="foot"><span>reports/onepager.md · README.md § Current state · reports/dataset_stats.md</span><span>6 / 6</span></div>
</section>
</body></html>"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    html = build_html()
    src = REPORTS / "vitsika_deck.html"
    src.write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": W, "height": H})
        page.goto(src.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(800)
        page.pdf(path=str(OUT_PDF), width=f"{W}px", height=f"{H}px", print_background=True, prefer_css_page_size=True)
        b.close()
    src.unlink()
    PREVIEWS.mkdir(exist_ok=True)
    for old in PREVIEWS.glob("slide-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", "72", str(OUT_PDF), str(PREVIEWS / "slide")], check=True)
    n = len(list(PREVIEWS.glob("slide-*.png")))
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} kB), {n} pages; previews in {PREVIEWS}/")


if __name__ == "__main__":
    main()
