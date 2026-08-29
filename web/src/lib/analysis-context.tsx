"use client";
/** Client-only state carrying the last /analyze answer (and the query image)
 * between Identify, /result and the atlas. It survives tab-surfing (React
 * state) and a reload (sessionStorage, image as a data URL when ≤ 2 MB).
 * It is cleared only by "← Identify another specimen" or a new upload. */
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import type { AnalyzeResponse, ExampleSpecimen } from "./api";

/** `truth` is set when the query came from the held-out picker: AntWeb's label.
 * `file` is the original upload (not persisted across reloads). */
export type Analysis = { result: AnalyzeResponse; imageUrl: string; filename: string; truth?: ExampleSpecimen; file?: File };
type Ctx = { analysis: Analysis | null; setAnalysis: (a: Analysis | null) => void; hydrated: boolean };

const KEY = "vitsika.analysis.v1";
const MAX_PERSISTED_IMAGE = 2 * 1024 * 1024;
type Stored = { result: AnalyzeResponse; filename: string; truth: ExampleSpecimen | null; image: string | null };

const AnalysisContext = createContext<Ctx>({ analysis: null, setAnalysis: () => {}, hydrated: false });

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [analysis, setState] = useState<Analysis | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) {
        const s = JSON.parse(raw) as Stored;
        setState({ result: s.result, imageUrl: s.image ?? "", filename: s.filename, truth: s.truth ?? undefined });
      }
    } catch { /* corrupt or unavailable storage: start empty */ }
    setHydrated(true);
  }, []);

  const setAnalysis = useCallback((a: Analysis | null) => {
    setState(a);
    try {
      if (!a) { sessionStorage.removeItem(KEY); return; }
      const save = (image: string | null) => {
        const s: Stored = { result: a.result, filename: a.filename, truth: a.truth ?? null, image };
        try { sessionStorage.setItem(KEY, JSON.stringify(s)); }
        catch { if (image) save(null); } // quota: keep the answer, drop the image
      };
      if (a.file && a.file.size <= MAX_PERSISTED_IMAGE) {
        const r = new FileReader();
        r.onload = () => save(typeof r.result === "string" ? r.result : null);
        r.onerror = () => save(null);
        r.readAsDataURL(a.file);
      } else save(null);
    } catch { /* storage unavailable */ }
  }, []);

  return <AnalysisContext.Provider value={{ analysis, setAnalysis, hydrated }}>{children}</AnalysisContext.Provider>;
}
export const useAnalysis = () => useContext(AnalysisContext);
