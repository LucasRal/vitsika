"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, imageUrl, type ExampleSpecimen } from "@/lib/api";
import { subfamilyColour } from "@/lib/palette";
import { Banner, Code, Genus, Spinner } from "@/components/ui";

type Props = { open: boolean; onClose: () => void; onPick: (s: ExampleSpecimen) => void; busy: boolean };

/** Modal grid of held-out TEST specimens (2 per genus). Labels can be hidden
 * so the visitor guesses before the model does. Hand-rolled: no UI library. */
export function ExamplePicker({ open, onClose, onPick, busy }: Props) {
  const [data, setData] = useState<ExampleSpecimen[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [hideLabels, setHideLabels] = useState(false);
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || data) return;
    api.examples(2).then((r) => setData(r.examples)).catch((e) => setError((e as Error).message));
  }, [open, data]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    dialog.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = (data ?? []).filter((s) => !q || s.genus.toLowerCase().includes(q) || s.subfamily.includes(q) || (s.species ?? "").toLowerCase().includes(q));
    const m = new Map<string, ExampleSpecimen[]>();
    rows.forEach((s) => m.set(s.subfamily, [...(m.get(s.subfamily) ?? []), s]));
    return Array.from(m.entries());
  }, [data, query]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose} role="presentation">
      <div ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="picker-title"
           className="hairline flex max-h-[88vh] w-full max-w-5xl flex-col rounded-md bg-surface shadow-xl outline-none"
           onClick={(e) => e.stopPropagation()}>
        <div className="hairline-b flex flex-wrap items-center gap-3 px-5 py-3">
          <div className="mr-auto">
            <h2 id="picker-title" className="text-lg">Pick a held-out specimen</h2>
            <p className="text-xs text-muted">
              {data ? `${data.length} of the 250 test images` : "Test images"}. The classifier was never fitted on these.
              Two per genus, chosen deterministically (not the easiest ones).
            </p>
          </div>
          <input className="input w-48" placeholder="Filter genus…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Filter examples" />
          <label className="flex items-center gap-1.5 text-sm text-ink-2">
            <input type="checkbox" checked={hideLabels} onChange={(e) => setHideLabels(e.target.checked)} /> hide labels (guess first)
          </label>
          <button className="btn" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="overflow-y-auto px-5 py-4">
          {error && <Banner tone="warning" title="Could not load the examples."><p>{error}</p></Banner>}
          {!data && !error && <Spinner label="Loading specimens…" />}
          {busy && <div className="mb-3"><Spinner label="Embedding with BioCLIP 2 (about a second)…" /></div>}
          {groups.map(([sub, rows]) => (
            <section key={sub} className="mb-5">
              <p className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: subfamilyColour(sub) }} aria-hidden />
                {hideLabels ? "subfamily hidden" : sub}
              </p>
              <ul className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
                {rows.map((s) => (
                  <li key={s.specimen_code}>
                    <button className="hairline group w-full rounded-md bg-surface p-1.5 text-left hover:border-accent disabled:opacity-50"
                            onClick={() => onPick(s)} disabled={busy} aria-label={hideLabels ? `specimen ${s.specimen_code}` : `${s.species ?? s.genus} ${s.specimen_code}`}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={imageUrl(s.specimen_code)} alt="" loading="lazy"
                           className="aspect-[4/3] w-full rounded-sm bg-surface-2 object-contain" />
                      <p className="mt-1 truncate text-xs leading-tight">
                        {hideLabels ? <span className="text-muted">? · guess</span> : <Genus name={s.species ?? s.genus} />}
                      </p>
                      <p className="text-[10px] text-muted"><Code code={s.specimen_code} /></p>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))}
          {data && groups.length === 0 && <p className="text-sm text-muted">No specimen matches.</p>}
          <p className="mt-2 text-[11px] text-muted">
            Images © their photographers via AntWeb (CC BY-SA). “Never seen” is true of the classifier; the BioCLIP 2 backbone
            was pretrained on public biology images and may have met some AntWeb photos.
          </p>
        </div>
      </div>
    </div>
  );
}
