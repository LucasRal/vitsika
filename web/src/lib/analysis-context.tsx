"use client";
/** Client-only state carrying the last /analyze answer (and the query image
 * as an object URL) from the Identify page to /result. Nothing is persisted:
 * a reload of /result sends the visitor back to Identify. */
import { createContext, useContext, useState, type ReactNode } from "react";
import type { AnalyzeResponse, ExampleSpecimen } from "./api";

/** `truth` is set when the query came from the held-out picker: AntWeb's label. */
export type Analysis = { result: AnalyzeResponse; imageUrl: string; filename: string; truth?: ExampleSpecimen };
type Ctx = { analysis: Analysis | null; setAnalysis: (a: Analysis | null) => void };

const AnalysisContext = createContext<Ctx>({ analysis: null, setAnalysis: () => {} });

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  return <AnalysisContext.Provider value={{ analysis, setAnalysis }}>{children}</AnalysisContext.Provider>;
}
export const useAnalysis = () => useContext(AnalysisContext);
