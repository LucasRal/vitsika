"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, type GenusInfo, type GeoResponse } from "@/lib/api";
import { num } from "@/lib/format";
import { subfamilyColour } from "@/lib/palette";
import { Banner, EffortNote, Genus, Spinner, Stat } from "@/components/ui";

// Leaflet touches `window` at import time: client-only, no SSR.
const GeoMap = dynamic(() => import("./GeoMap").then((m) => m.GeoMap), {
  ssr: false, loading: () => <div className="hairline flex h-[520px] items-center justify-center rounded-md bg-surface-2"><Spinner label="Loading map…" /></div>,
});

export function DistributionClient({ genera }: { genera: GenusInfo[] }) {
  const params = useSearchParams();
  const router = useRouter();
  const requested = params.get("genus");
  const known = genera.some((g) => g.name === requested);
  const genus = known ? requested! : genera[0]?.name;
  const [geo, setGeo] = useState<GeoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!genus) return;
    let cancelled = false;
    setGeo(null); setError(null);
    api.geo(genus).then((g) => { if (!cancelled) setGeo(g); })
                  .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [genus]);

  const select = (name: string) => router.replace(`/distribution?genus=${encodeURIComponent(name)}`);
  const provinces = geo ? Object.entries(geo.provinces).sort((a, b) => b[1] - a[1]) : [];
  const maxProv = provinces[0]?.[1] ?? 1;
  const colour = subfamilyColour(geo?.subfamily ?? genera.find((g) => g.name === genus)?.subfamily ?? "");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm text-ink-2" htmlFor="genus-select">Genus</label>
        <select id="genus-select" className="input genus" value={genus} onChange={(e) => select(e.target.value)}>
          {genera.map((g) => <option key={g.name} value={g.name}>{g.name} ({g.subfamily})</option>)}
        </select>
        {requested && !known && <span className="text-xs text-warning">“{requested}” is not one of the 27 genera; showing <Genus name={genus} />.</span>}
      </div>
      {error && <Banner tone="warning" title="Could not load the distribution."><p>{error}</p></Banner>}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="space-y-2">
          {geo ? <GeoMap points={geo.points} colour={colour} genus={geo.genus} />
               : !error && <div className="hairline flex h-[520px] items-center justify-center rounded-md bg-surface-2"><Spinner label={`Loading ${genus}…`} /></div>}
          <EffortNote />
          {geo?.points_capped && <p className="text-xs text-muted">Showing a seeded subsample of 1,000 of {num(geo.n_with_coords)} georeferenced specimens.</p>}
        </div>
        <div className="space-y-4">
          {geo && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Specimens" value={num(geo.n_specimens)} sub={`${num(geo.n_with_coords)} with coordinates`} />
                <Stat label="Named species" value={num(geo.n_species)} sub={geo.n_unidentified ? `${num(geo.n_unidentified)} identified to genus only` : "all identified to species"} />
                <Stat label="Elevation" value={geo.elevation.median === null ? "—" : `${Math.round(geo.elevation.median)} m`}
                      sub={geo.elevation.n ? `${Math.round(geo.elevation.min!)}–${Math.round(geo.elevation.max!)} m · median of ${num(geo.elevation.n)}` : "no elevation data"} />
                <Stat label="Collected" value={geo.year_min === null ? "—" : geo.year_min === geo.year_max ? geo.year_min : `${geo.year_min}–${geo.year_max}`} sub="year range of dated records" />
              </div>
              <div>
                <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted">Specimens by province</p>
                <ul className="space-y-1.5">
                  {provinces.map(([p, n]) => (
                    <li key={p} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2 text-sm">
                      <span className={p === "nan" ? "text-muted" : ""}>{p === "nan" ? "(no province)" : p}</span>
                      <div className="h-1.5 rounded-sm bg-surface-2"><div className="h-full rounded-sm" style={{ width: `${(n / maxProv) * 100}%`, background: colour }} /></div>
                      <span className="code text-right">{n}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
