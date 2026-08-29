/** Three-tier confidence verdict for an /analyze answer.
 *
 * The probe's softmax alone is a poor gauge: a top-1 of 37 % reads alarming
 * even when every nearest reference specimen agrees with it. So the verdict
 * combines the classifier (top-1 probability) with the retrieval evidence
 * (how many of the k nearest training specimens share the top-1 genus at a
 * high cosine similarity). Pure function; unit-tested in verdict.test.ts. */
import type { AnalyzeResponse } from "./api";

export const LOW_CONFIDENCE = 0.5;   // top-1 below this → the classifier is unsure
export const SUPPORT_SIMILARITY = 0.85; // a neighbour counts as support only above this cosine
export const SUPPORT_MIN = 3;        // ≥ this many supporting neighbours rescues a modest top-1

export type Tier = "confident" | "supported" | "low";
export type Verdict = {
  tier: Tier;
  genus: string;
  top1: number;
  /** neighbours with genus == top-1 genus and similarity >= SUPPORT_SIMILARITY */
  support: number;
  nSimilar: number;
};

export function verdict(r: Pick<AnalyzeResponse, "predictions" | "similar">): Verdict {
  const top = r.predictions[0];
  const top1 = top?.probability ?? 0;
  const genus = top?.genus ?? "";
  const support = r.similar.filter((s) => s.genus === genus && s.similarity >= SUPPORT_SIMILARITY).length;
  const tier: Tier = top1 >= LOW_CONFIDENCE ? "confident" : support >= SUPPORT_MIN ? "supported" : "low";
  return { tier, genus, top1, support, nSimilar: r.similar.length };
}
