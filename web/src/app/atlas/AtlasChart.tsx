"use client";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { AtlasPoint } from "@/lib/api";
import { antwebUrl, imageUrl } from "@/lib/api";
import { OTHER, PALETTE, SUBFAMILY_ORDER } from "@/lib/palette";
import { Genus } from "@/components/ui";

type Mode = "subfamily" | "genera" | "one";
const N_TOP_GENERA = 8;

/** Convergence islands, medians looked up in data/umap_coords.csv at build time
 * (island A: 47 points within r=1.2, Aphaenogaster 25 / Odontomachus 13 / C. imitator 7;
 *  island B: 32 points, Anochetus 27 + C. reaumuri 3 + Tetraponera grandidieri 2). */
const ANNOTATIONS = [
  { x: 3.51, y: 9.23, title: "Long-legged island", text: "Aphaenogaster, Odontomachus and Camponotus imitator — three subfamilies sharing one body plan" },
  { x: 5.48, y: 9.53, title: "Anochetus island", text: "Anochetus with a few Camponotus reaumuri and Tetraponera grandidieri" },
];
/** ECharts draws on canvas, where CSS custom properties are not resolved:
 * read the design tokens off :root and re-read when the colour scheme flips. */
type Tokens = { ink: string; ink2: string; muted: string; hairline: string; accent: string; surface: string; surface2: string };
function useTokens(): Tokens {
  const read = (): Tokens => {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
    return { ink: v("--ink", "#141414"), ink2: v("--ink-2", "#4f4e4a"), muted: v("--muted", "#8a8880"), hairline: v("--hairline", "#ccc"),
             accent: v("--accent", "#534ab7"), surface: v("--surface", "#fff"), surface2: v("--surface-2", "#f8f8f6") };
  };
  const [tokens, setTokens] = useState<Tokens>({ ink: "#141414", ink2: "#4f4e4a", muted: "#8a8880", hairline: "rgba(20,20,20,0.18)", accent: "#534ab7", surface: "#ffffff", surface2: "#f8f8f6" });
  useEffect(() => {
    setTokens(read());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setTokens(read());
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return tokens;
}

const STAR = "path://M12 2l2.9 6.6 7.1.6-5.4 4.7 1.6 7.1L12 17.3 5.8 21l1.6-7.1L2 9.2l7.1-.6z";

export function AtlasChart({ points }: { points: AtlasPoint[] }) {
  const params = useSearchParams();
  const star = useMemo(() => params.get("star") === "1" && params.get("x") && params.get("y")
    ? { x: Number(params.get("x")), y: Number(params.get("y")), label: params.get("label") ?? "Your upload" } : null, [params]);
  const [mode, setMode] = useState<Mode>("subfamily");
  const [query, setQuery] = useState("");
  const [one, setOne] = useState<string | null>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const t = useTokens();

  const genera = useMemo(() => {
    const counts = new Map<string, number>();
    points.forEach((p) => counts.set(p.genus, (counts.get(p.genus) ?? 0) + 1));
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [points]);
  const matches = useMemo(() => genera.filter(([g]) => g.toLowerCase().includes(query.toLowerCase())), [genera, query]);

  const option = useMemo<EChartsOption>(() => {
    // categories -> (name, colour, order) — fixed palette order, never cycled; 9th+ folds to "other"
    let categories: { name: string; colour: string; keyOf: (p: AtlasPoint) => string }[];
    if (mode === "subfamily") {
      const present = SUBFAMILY_ORDER.filter((s) => points.some((p) => p.subfamily === s));
      categories = present.map((s, i) => ({ name: s, colour: PALETTE[i] ?? OTHER, keyOf: (p) => p.subfamily }));
    } else if (mode === "genera") {
      const top = genera.slice(0, N_TOP_GENERA).map(([g]) => g);
      categories = [...top.map((g, i) => ({ name: g, colour: PALETTE[i], keyOf: (p: AtlasPoint) => (top.includes(p.genus) ? p.genus : "other") })),
                    { name: "other", colour: OTHER, keyOf: (p: AtlasPoint) => (top.includes(p.genus) ? p.genus : "other") }];
    } else {
      categories = [{ name: "other", colour: OTHER, keyOf: (p) => (p.genus === one ? one : "other") },
                    ...(one ? [{ name: one, colour: PALETTE[0], keyOf: (p: AtlasPoint) => (p.genus === one ? one : "other") }] : [])];
    }
    const series: NonNullable<EChartsOption["series"]> = categories.map((c) => ({
      name: c.name, type: "scatter", symbolSize: c.name === "other" ? 5 : 7,
      itemStyle: { color: c.colour, opacity: c.name === "other" ? 0.45 : 0.85, borderColor: t.surface, borderWidth: 0.5 },
      emphasis: { scale: 1.8 },
      data: points.filter((p) => c.keyOf(p) === c.name).map((p) => ({ value: [p.x, p.y], p })),
    }));
    if (showAnnotations) series.push({
      name: "annotation", type: "scatter", silent: true, symbol: "circle", symbolSize: 58, z: 1,
      itemStyle: { color: "transparent", borderColor: t.ink, borderWidth: 0.8, borderType: "dashed" },
      label: { show: true, position: "top", distance: 8, color: t.ink, fontFamily: "Source Serif 4, Georgia, serif", fontSize: 12, fontStyle: "italic",
               formatter: (d) => (d.data as { title: string }).title },
      data: ANNOTATIONS.map((a) => ({ value: [a.x, a.y], ...a })),
    });
    if (star) series.push({
      name: "your upload", type: "scatter", symbol: STAR, symbolSize: 26, z: 10,
      itemStyle: { color: t.accent, borderColor: t.surface, borderWidth: 1 },
      label: { show: true, position: "right", distance: 10, color: t.accent, fontWeight: 600, fontSize: 12, formatter: star.label,
               backgroundColor: t.surface, borderColor: t.hairline, borderWidth: 0.5, padding: [2, 6], borderRadius: 3 },
      data: [{ value: [star.x, star.y] }],
    });
    return {
      animation: false, backgroundColor: "transparent",
      legend: { type: "scroll", bottom: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: t.ink2, fontSize: 12 }, pageTextStyle: { color: t.muted },
                data: categories.map((c) => c.name).filter((n) => !(mode === "one" && n === "other")) },
      grid: { left: 24, right: 24, top: 24, bottom: 48 },
      // no units on UMAP axes: no ticks, no labels, no zero lines
      xAxis: { type: "value", scale: true, axisLabel: { show: false }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, axisLabel: { show: false }, axisTick: { show: false }, axisLine: { show: false }, splitLine: { show: false } },
      dataZoom: [{ type: "inside", xAxisIndex: 0, filterMode: "none" }, { type: "inside", yAxisIndex: 0, filterMode: "none" }],
      tooltip: {
        trigger: "item", backgroundColor: t.surface, borderColor: t.hairline, borderWidth: 0.5, padding: 8,
        textStyle: { color: t.ink, fontSize: 12 }, enterable: false, confine: true,
        formatter: (raw) => {
          const d = (Array.isArray(raw) ? raw[0] : raw).data as { p?: AtlasPoint; title?: string; text?: string };
          if (d.title) return `<b>${d.title}</b><br/><span style="color:var(--ink-2)">${d.text}</span>`;
          if (!d.p) return "";
          const p = d.p;
          const thumb = p.image_available
            ? `<img src="${imageUrl(p.specimen_code)}" width="160" height="120" style="display:block;object-fit:contain;background:var(--surface-2);border-radius:3px" onerror="this.outerHTML='<div style=\\'width:160px;height:120px;display:flex;align-items:center;justify-content:center;color:var(--muted);background:var(--surface-2)\\'>no image</div>'" />`
            : `<div style="width:160px;height:120px;display:flex;align-items:center;justify-content:center;color:var(--muted);background:var(--surface-2)">no image</div>`;
          return `${thumb}<div style="margin-top:6px"><i>${p.species ?? p.genus + " sp."}</i></div>` +
                 `<div style="color:var(--ink-2)">${p.subfamily}</div>` +
                 `<div style="font-family:var(--font-mono);font-size:11px;color:var(--muted)">${p.specimen_code} · click for AntWeb</div>`;
        },
      },
      series,
    };
  }, [points, genera, mode, one, star, showAnnotations, t]);

  const onClick = (e: { data?: { p?: AtlasPoint } }) => { if (e.data?.p) window.open(antwebUrl(e.data.p.specimen_code), "_blank", "noopener"); };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-ink-2">Colour by</span>
        {(["subfamily", "genera", "one"] as Mode[]).map((m) => (
          <button key={m} className={`btn ${mode === m ? "btn-primary" : ""}`} onClick={() => setMode(m)} aria-pressed={mode === m}>
            {m === "subfamily" ? "subfamily" : m === "genera" ? `${N_TOP_GENERA} largest genera` : "one genus"}
          </button>
        ))}
        {mode === "one" && (
          <div className="relative">
            <input className="input w-56" placeholder="Type a genus…" value={one ?? query} aria-label="Highlight one genus"
                   onChange={(e) => { setOne(null); setQuery(e.target.value); }} onFocus={() => { if (one) { setQuery(one); setOne(null); } }} />
            {!one && (
              <ul className="hairline absolute z-20 mt-1 max-h-64 w-56 overflow-auto rounded-md bg-surface py-1 shadow-sm" role="listbox">
                {matches.map(([g, n]) => (
                  <li key={g}><button className="flex w-full justify-between px-3 py-1 text-left hover:bg-surface-2" onClick={() => { setOne(g); setQuery(""); }} role="option" aria-selected={false}>
                    <Genus name={g} /><span className="code text-muted">{n}</span></button></li>
                ))}
                {matches.length === 0 && <li className="px-3 py-1 text-muted">no match</li>}
              </ul>
            )}
          </div>
        )}
        <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-2">
          <input type="checkbox" checked={showAnnotations} onChange={(e) => setShowAnnotations(e.target.checked)} /> island annotations
        </label>
      </div>
      {star && <p className="text-xs text-accent">★ {star.label} — placed with the fitted UMAP’s transform (approximate: a training image lands near, not on, its own point).</p>}
      <div className="hairline rounded-md bg-surface p-2">
        <ReactECharts option={option} style={{ height: 640 }} notMerge onEvents={{ click: onClick }} opts={{ renderer: "canvas" }} />
      </div>
      <p className="text-xs text-muted">Scroll to zoom, drag to pan. The two dashed rings mark cross-subfamily islands where the model groups ants by body plan rather than lineage — see Methods.</p>
    </div>
  );
}
