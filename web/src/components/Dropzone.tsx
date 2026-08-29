"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { api, ApiError, imageUrl, type ExampleSpecimen } from "@/lib/api";
import { useAnalysis, type Analysis } from "@/lib/analysis-context";
import { Banner, Genus } from "@/components/ui";
import { pct } from "@/lib/format";
import { ExamplePicker } from "@/components/ExamplePicker";
import { AnalyzeProgress, type Stage } from "@/components/AnalyzeProgress";

const MAX_MB = 10;
const EMBED_TEXT_AFTER_MS = 300; // "Uploading…" → "Embedding…" once the request is in flight
const EXAMPLE = {
  url: "/examples/casent0002219_p.jpg", name: "casent0002219_p.jpg",
  truth: { specimen_code: "casent0002219", genus: "Royidris", subfamily: "myrmicinae", species: "Royidris notorthotenes",
           photographer: "April Nobile", license: "CC BY-SA", image_url: "", antweb_url: "https://www.antweb.org/specimen/casent0002219" } as ExampleSpecimen,
};

type Props = {
  /** Called with the finished analysis (already stored in the AnalysisContext).
   * Default: navigate to /result. */
  onDone?: (a: Analysis) => void;
  /** Smaller box and shorter copy — used inside the atlas card. */
  compact?: boolean;
  submitLabel?: string;
};

/** Upload box + example buttons + the inline progress/error panel that takes
 * its place while /analyze runs. Shared by Identify and the Atlas card. */
export function Dropzone({ onDone, compact = false, submitLabel = "Identify genus" }: Props) {
  const router = useRouter();
  const params = useSearchParams();
  const { analysis, setAnalysis, hydrated } = useAnalysis();
  const input = useRef<HTMLInputElement>(null);
  const zone = useRef<HTMLDivElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const last = useRef<{ file: File; truth?: ExampleSpecimen } | null>(null);
  // The previous answer shown in the box when coming back to Identify (null once a new photo is chosen).
  const [restored, setRestored] = useState<Analysis | null>(null);
  const timer = useRef<number | null>(null);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);

  /** Forget the previous answer, start from an empty box and focus it. */
  const reset = useCallback(() => {
    if (analysis?.imageUrl.startsWith("blob:")) URL.revokeObjectURL(analysis.imageUrl);
    setAnalysis(null);
    setRestored(null); setFile(null); setPreview(null); setError(null); setStage("idle");
    requestAnimationFrame(() => zone.current?.focus());
  }, [analysis, setAnalysis]);

  // "← Identify another specimen" arrives as /?new=1.
  const fresh = params.get("new") === "1";
  useEffect(() => {
    if (!fresh) return;
    reset();
    router.replace("/", { scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fresh]);

  // Coming back to Identify: keep the last result in the box until the visitor starts over.
  useEffect(() => {
    if (compact || fresh || !hydrated || !analysis || file || restored) return;
    setRestored(analysis);
    setFile(analysis.file ?? null);
    setPreview(analysis.imageUrl || null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, analysis, fresh, compact]);

  const busy = stage === "loading" || stage === "uploading" || stage === "embedding";

  const pick = useCallback((f: File | null) => {
    setError(null);
    if (!f) return;
    setRestored(null);
    if (!f.type.startsWith("image/")) return setError(`“${f.name}” is not an image (${f.type || "unknown type"}).`);
    if (f.size > MAX_MB * 1024 * 1024) return setError(`“${f.name}” is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_MB} MB.`);
    setFile(f);
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
  }, []);

  const submit = async (f: File, truth?: ExampleSpecimen) => {
    last.current = { file: f, truth };
    setError(null);
    setStage("uploading");
    timer.current = window.setTimeout(() => setStage("embedding"), EMBED_TEXT_AFTER_MS);
    try {
      const result = await api.analyze(f);
      const a: Analysis = { result, imageUrl: URL.createObjectURL(f), filename: f.name, truth, file: f };
      setAnalysis(a);
      setStage("done");
      if (onDone) onDone(a); else router.push("/result");
    } catch (e) {
      setError(e instanceof ApiError && e.status === 0
        ? "The analysis service is not reachable. Is the API running?" : (e as Error).message);
      setStage("error");
    } finally {
      if (timer.current) { window.clearTimeout(timer.current); timer.current = null; }
    }
  };

  const retry = () => { if (last.current) submit(last.current.file, last.current.truth); };
  const cancel = () => { setStage("idle"); setError(null); };

  /** Fetch an example's bytes (local asset or the API's /images), then analyse. */
  const fetchAndSubmit = async (url: string, name: string, truth: ExampleSpecimen) => {
    setStage("loading");
    setError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`image ${name}: HTTP ${res.status}`);
      const f = new File([await res.blob()], name, { type: "image/jpeg" });
      pick(f);
      await submit(f, truth);
    } catch (e) {
      last.current = null;
      setError((e as Error).message);
      setStage("error");
    }
  };
  const useExample = () => fetchAndSubmit(EXAMPLE.url, EXAMPLE.name, EXAMPLE.truth);
  const usePicked = (s: ExampleSpecimen) => { setPickerOpen(false); return fetchAndSubmit(imageUrl(s.specimen_code), `${s.specimen_code}_p.jpg`, s); };

  return (
    <div className="space-y-3">
      {stage !== "idle" ? (
        <AnalyzeProgress stage={stage} preview={preview} filename={file?.name} error={error} compact={compact}
                         onRetry={last.current ? retry : cancel} onCancel={cancel} />
      ) : (
        <div
          ref={zone} role="button" tabIndex={0} aria-label="Upload an ant photo"
          onClick={() => input.current?.click()}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0] ?? null); }}
          className={`dropzone flex ${compact ? "min-h-40" : "min-h-56"} cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed p-6 text-center transition-colors
            ${drag ? "border-accent bg-accent-soft" : "border-hairline bg-surface-2 hover:border-accent"}`}
        >
          <input ref={input} type="file" accept="image/*" className="hidden"
                 onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="Selected specimen" className={`${compact ? "max-h-40" : "max-h-64"} rounded-sm object-contain`} />
          ) : restored ? (
            <p className="text-sm text-muted">Previous photo not kept (too large to store) — the result is still available.</p>
          ) : (
            <>
              <p className={`font-serif ${compact ? "text-base" : "text-lg"}`}>Drop a profile-view photo of an ant</p>
              <p className="text-sm text-muted">or click to choose a file · JPEG/PNG, up to {MAX_MB} MB</p>
              {!compact && <p className="text-xs text-muted">Best results: lateral (side) view of a worker on a plain background, as on AntWeb.</p>}
            </>
          )}
        </div>
      )}
      {restored && stage === "idle" ? (
        <p className="text-sm" role="status" data-testid="last-result">
          <span className="text-muted">Last result:</span>{" "}
          <Genus name={restored.result.predictions[0].genus} className="font-medium" />{" "}
          <span className="text-muted">{pct(restored.result.predictions[0].probability)}</span>
          <span className="text-muted"> · {restored.filename}</span>
        </p>
      ) : (
        file && stage === "idle" && <p className="text-xs text-muted">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>
      )}
      {error && stage === "idle" && <Banner tone="warning" title={error} />}
      <div className="flex flex-wrap items-center gap-2">
        {restored && stage === "idle" ? (
          <>
            <Link className="btn btn-primary" href="/result">View result →</Link>
            <button className="btn" onClick={reset}><span aria-hidden>←</span> Identify another specimen</button>
          </>
        ) : (
          <button className="btn btn-primary" disabled={!file || busy || stage === "error"} onClick={() => file && submit(file)}>{submitLabel}</button>
        )}
        <button className="btn" disabled={busy} onClick={() => setPickerOpen(true)}>Pick a held-out specimen…</button>
        <button className="btn" disabled={busy} onClick={useExample}>Quick example</button>
        {file && !restored && stage === "idle" && <button className="btn" onClick={() => { setFile(null); setPreview(null); }}>Clear</button>}
      </div>
      {!compact && (
        <p className="text-xs text-muted">
          Held-out specimens are test images the classifier was never fitted on; the quick example is{" "}
          <span className="genus">Royidris notorthotenes</span> <span className="code">casent0002219</span> (© April Nobile, AntWeb).
        </p>
      )}
      <ExamplePicker open={pickerOpen} onClose={() => setPickerOpen(false)} onPick={usePicked} busy={busy} />
    </div>
  );
}
