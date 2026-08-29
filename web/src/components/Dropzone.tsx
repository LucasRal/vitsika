"use client";
import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError, imageUrl, type ExampleSpecimen } from "@/lib/api";
import { useAnalysis } from "@/lib/analysis-context";
import { Banner, Spinner } from "@/components/ui";
import { ExamplePicker } from "@/components/ExamplePicker";

const MAX_MB = 10;
const EXAMPLE = {
  url: "/examples/casent0002219_p.jpg", name: "casent0002219_p.jpg",
  truth: { specimen_code: "casent0002219", genus: "Royidris", subfamily: "myrmicinae", species: "Royidris notorthotenes",
           photographer: "April Nobile", license: "CC BY-SA", image_url: "", antweb_url: "https://www.antweb.org/specimen/casent0002219" } as ExampleSpecimen,
};

export function Dropzone() {
  const router = useRouter();
  const { setAnalysis } = useAnalysis();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);

  const pick = useCallback((f: File | null) => {
    setError(null);
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError(`“${f.name}” is not an image (${f.type || "unknown type"}).`);
    if (f.size > MAX_MB * 1024 * 1024) return setError(`“${f.name}” is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_MB} MB.`);
    setFile(f);
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
  }, []);

  const submit = async (f: File, truth?: ExampleSpecimen) => {
    setBusy("Embedding with BioCLIP 2 on CPU — about a second…");
    setError(null);
    try {
      const result = await api.analyze(f);
      setAnalysis({ result, imageUrl: URL.createObjectURL(f), filename: f.name, truth });
      router.push("/result");
    } catch (e) {
      const msg = e instanceof ApiError && e.status === 0
        ? "The analysis service is not reachable. Is the API running?" : (e as Error).message;
      setError(msg);
      setBusy(null);
    }
  };

  const useExample = async () => {
    setBusy("Loading the example specimen…");
    try {
      const blob = await (await fetch(EXAMPLE.url)).blob();
      const f = new File([blob], EXAMPLE.name, { type: "image/jpeg" });
      pick(f);
      await submit(f, EXAMPLE.truth);
    } catch (e) { setError((e as Error).message); setBusy(null); }
  };

  /** A held-out test specimen from the picker: fetch its jpg from the API, then analyse it. */
  const usePicked = async (s: ExampleSpecimen) => {
    setBusy(`Loading ${s.specimen_code}…`);
    try {
      const res = await fetch(imageUrl(s.specimen_code));
      if (!res.ok) throw new Error(`image ${s.specimen_code}: HTTP ${res.status}`);
      const f = new File([await res.blob()], `${s.specimen_code}_p.jpg`, { type: "image/jpeg" });
      pick(f);
      setPickerOpen(false);
      await submit(f, s);
    } catch (e) { setError((e as Error).message); setBusy(null); }
  };

  return (
    <div className="space-y-3">
      <div
        role="button" tabIndex={0} aria-label="Upload an ant photo"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0] ?? null); }}
        className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed p-6 text-center transition-colors
          ${drag ? "border-accent bg-accent-soft" : "border-hairline bg-surface-2 hover:border-accent"}`}
      >
        <input ref={input} type="file" accept="image/*" className="hidden"
               onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="Selected specimen" className="max-h-64 rounded-sm object-contain" />
        ) : (
          <>
            <p className="font-serif text-lg">Drop a profile-view photo of an ant</p>
            <p className="text-sm text-muted">or click to choose a file · JPEG/PNG, up to {MAX_MB} MB</p>
            <p className="text-xs text-muted">Best results: lateral (side) view of a worker on a plain background, as on AntWeb.</p>
          </>
        )}
      </div>
      {file && <p className="text-xs text-muted">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
      {error && <Banner tone="warning" title={error} />}
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" disabled={!file || !!busy} onClick={() => file && submit(file)}>Identify genus</button>
        <button className="btn" disabled={!!busy} onClick={() => setPickerOpen(true)}>Pick a held-out specimen…</button>
        <button className="btn" disabled={!!busy} onClick={useExample}>Quick example</button>
        {file && !busy && <button className="btn" onClick={() => { setFile(null); setPreview(null); }}>Clear</button>}
        {busy && <Spinner label={busy} />}
      </div>
      <p className="text-xs text-muted">
        Held-out specimens are test images the classifier was never fitted on; the quick example is{" "}
        <span className="genus">Royidris notorthotenes</span> <span className="code">casent0002219</span> (© April Nobile, AntWeb).
      </p>
      <ExamplePicker open={pickerOpen} onClose={() => setPickerOpen(false)} onPick={usePicked} busy={!!busy} />
    </div>
  );
}
