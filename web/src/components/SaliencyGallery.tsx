"use client";
import { useEffect, useState } from "react";
import Image from "next/image";
import { antwebUrl, apiBase } from "@/lib/api";
import { Genus } from "@/components/ui";
import { SaliencyChat } from "@/components/SaliencyChat";

export type RiseRun = {
  code: string;
  trueGenus: string;
  species: string;
  target: string;
  pFull: number;
  caption: string;
  creator: string;
  /** mean of the saliency map, the zero of the shared deviation scale */
  baseline: number;
};

type Scale = "per-map" | "shared";

const OPTIONS: [Scale, string][] = [["per-map", "Per-map scale"], ["shared", "Shared scale"]];

export function SaliencyGallery({ runs }: { runs: RiseRun[] }) {
  const [scale, setScale] = useState<Scale>("per-map");
  // "Discuss this map" exists only when the API has an LLM configured
  const [chat, setChat] = useState(false);
  useEffect(() => {
    const ctrl = new AbortController();
    fetch(`${apiBase()}/saliency/chat/status`, { signal: ctrl.signal, cache: "no-store" })
      .then((r) => (r.ok ? r.json() : { enabled: false }))
      .then((s: { enabled: boolean }) => setChat(Boolean(s.enabled)))
      .catch(() => setChat(false));
    return () => ctrl.abort();
  }, []);
  return (
    <section className="space-y-6">
      <div className="space-y-2">
        <div role="radiogroup" aria-label="Colour scale" className="hairline inline-flex rounded-sm p-0.5 text-sm">
          {OPTIONS.map(([value, label]) => {
            const active = scale === value;
            return (
              <button key={value} type="button" role="radio" aria-checked={active}
                onClick={() => setScale(value)}
                className={`rounded-sm px-2.5 py-1 ${active ? "bg-accent-soft text-accent" : "text-ink-2 hover:bg-surface-2"}`}>
                {label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-muted">
          Per-map: each map stretched to its own range — shows structure. Shared: same deviation scale for all maps —
          shows which maps actually carry signal.
        </p>
      </div>

      {runs.map((r) => {
        const useShared = scale === "shared";
        const overlay = `/rise/${r.code}_${r.target}_overlay${useShared ? "_shared" : ""}.png`;
        return (
          <div
            key={`${r.code}-${r.target}`}
            className="hairline grid grid-cols-1 gap-4 rounded-md bg-surface-2 p-4 sm:grid-cols-[1fr_1fr_1.1fr] sm:items-center"
          >
            <figure>
              <Image
                src={`/rise/${r.code}_photo.png`}
                alt={`${r.trueGenus} specimen ${r.code}, profile view`}
                width={462}
                height={462}
                className="hairline w-full rounded-sm bg-white"
              />
              <figcaption className="mt-1 text-xs text-muted">
                <Genus name={r.trueGenus} /> <span className="code">{r.code}</span> · ©{" "}
                {r.creator} ·{" "}
                <a className="link" href={antwebUrl(r.code)} target="_blank" rel="noreferrer">
                  AntWeb
                </a>{" "}
                (CC BY-SA)
              </figcaption>
            </figure>
            <figure>
              <Image
                src={overlay}
                alt={`RISE saliency for ${r.target} on specimen ${r.code}${useShared ? ", shared colour scale" : ""}`}
                width={useShared ? 529 : 516}
                height={useShared ? 565 : 544}
                className="hairline w-full rounded-sm bg-white"
              />
              <figcaption className="mt-1 text-xs text-muted">
                RISE for <Genus name={r.target} /> · full-image P = {r.pFull.toFixed(3)}
                {useShared && ` · baseline P = ${r.baseline.toFixed(3)}`}
              </figcaption>
            </figure>
            <p className="text-sm text-ink-2">{r.caption}</p>
            {chat && (
              <div className="sm:col-span-3">
                <SaliencyChat code={r.code} target={r.target} />
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
